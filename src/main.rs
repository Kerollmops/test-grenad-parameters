use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Cursor, ErrorKind, Read};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::{iter, str};

use anyhow::Context;
use clap::Parser;
use gabble::Gabble;
use grenad::{CompressionType, Reader, ReaderCursor, WriterBuilder};
use heed::{Database, Env, EnvOpenOptions, RoTxn};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use roaring::RoaringBitmap;

const FIVE_GIB: usize = 5 * 1024 * 1024 * 1024;
const MAX_BITMAP_LEN: usize = 116_000_000;
const POSSIBLE_READ_METHODS: &[&str] =
    &["direct", "read-to-vec", "bufreader", "memory-mapped", "memory-mapped-bufreader"];
const POSSIBLE_SORT_METHODS: &[&str] = &["iter-only", "iter-and-jump", "jump-only"];

#[derive(Parser)]
#[clap(version = "1.0", author = "Kevin K. <kbknapp@gmail.com>")]
struct Opts {
    /// A level of verbosity, and can be used multiple times
    #[clap(short, long, parse(from_occurrences))]
    verbose: i32,

    /// The folder where to create the grenad databases.
    #[clap(long)]
    folder: PathBuf,

    #[clap(subcommand)]
    subcommand: SubCommand,
}

#[derive(Parser)]
enum SubCommand {
    /// Run the extended test suite which consist in generating a big list of
    /// key-value pairs, storing them with different parameters and executing
    /// a full iteration followed by random jumps over the list of entries.
    ///
    /// The generated RoaringBitmaps are deserialized in the process.
    ExtendedRandomTests {
        #[clap(long, default_value = "42")]
        seed: u64,

        #[clap(long, default_value = "10000")]
        entry_count: u64,

        #[clap(
            long,
            default_value = "direct",
            possible_values = POSSIBLE_READ_METHODS,
        )]
        read_method: String,

        #[clap(
            long,
            default_value = "jump-only",
            possible_values = POSSIBLE_SORT_METHODS,
        )]
        sort_by: String,
    },
    /// Run the extended test suite which consist in retrieving a big list of
    /// key-value pairs from a provided file, storing them with different parameters
    /// and executing a full iteration followed by random jumps over the list of entries.
    ///
    /// The generated RoaringBitmaps are deserialized in the process.
    ExtendedTests {
        #[clap(long, default_value = "42")]
        seed: u64,

        /// The grenad file to read entries from to execute the extended suite of tests.
        #[clap(long)]
        file: PathBuf,

        #[clap(
            long,
            default_value = "direct",
            possible_values = POSSIBLE_READ_METHODS,
        )]
        read_method: String,

        #[clap(
            long,
            default_value = "jump-only",
            possible_values = POSSIBLE_SORT_METHODS,
        )]
        sort_by: String,
    },
    OneTest {
        #[clap(long, default_value = "42")]
        seed: u64,

        /// The grenad file to read entries from to execute the extended suite of tests.
        #[clap(long)]
        file: PathBuf,

        #[clap(
            long,
            default_value = "direct",
            possible_values = POSSIBLE_READ_METHODS,
        )]
        read_method: String,

        #[clap(long)]
        compression: Option<CompressionType>,

        #[clap(long)]
        index_levels: u8,

        #[clap(long)]
        block_size: usize,

        #[clap(long)]
        index_key_interval: NonZeroUsize,
    },
    OneRandomTest {
        #[clap(long, default_value = "42")]
        seed: u64,

        #[clap(long, default_value = "10000")]
        entry_count: u64,

        #[clap(
            long,
            default_value = "direct",
            possible_values = POSSIBLE_READ_METHODS,
        )]
        read_method: String,

        #[clap(long)]
        compression: Option<CompressionType>,

        #[clap(long)]
        index_levels: u8,

        #[clap(long)]
        block_size: usize,

        #[clap(long)]
        index_key_interval: NonZeroUsize,
    },
    OneLmdbTest {
        #[clap(long, default_value = "42")]
        seed: u64,

        /// The grenad file to read entries from to execute the extended suite of tests.
        #[clap(long)]
        file: PathBuf,
    },
    OneRandomLmdbTest {
        #[clap(long, default_value = "42")]
        seed: u64,

        #[clap(long, default_value = "10000")]
        entry_count: u64,
    },
}

#[derive(Debug, Copy, Clone)]
struct Parameters {
    compression: CompressionType,
    index_levels: u8,
    block_size: usize,
    index_key_interval: NonZeroUsize,
}

#[derive(Debug, Copy, Clone)]
struct Results {
    iter_time: Duration,
    jump_time: Duration,
}

fn main() -> anyhow::Result<()> {
    let Opts { verbose, folder, subcommand } = Opts::try_parse()?;

    match subcommand {
        SubCommand::ExtendedRandomTests { seed, entry_count, read_method, sort_by } => {
            println!("generating random words...");
            let mut rng = StdRng::seed_from_u64(seed);
            let pb = ProgressBar::new(entry_count)
                .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));
            let mut words: Vec<_> = iter::repeat_with(|| {
                Gabble::new().with_length(rng.gen_range(3..=15)).generate(&mut rng)
            })
            .take(entry_count as usize)
            .progress_with(pb)
            .collect();
            words.sort_unstable();
            words.dedup();
            println!("{} unique words generated!", words.len());

            let compressions =
                vec![CompressionType::None, CompressionType::Snappy, CompressionType::Lz4];
            let index_levels = vec![0, 1, 2, 3];
            let block_sizes = vec![8 * 1024, 4 * 1024, 2 * 1024, 1 * 1024, 512];
            let index_key_intervals: Vec<_> =
                vec![32, 24, 16, 12, 8, 4, 2].into_iter().filter_map(NonZeroUsize::new).collect();

            let mut parameters = Vec::new();
            for &compression in &compressions {
                for &index_levels in &index_levels {
                    for &block_size in &block_sizes {
                        for &index_key_interval in &index_key_intervals {
                            parameters.push(Parameters {
                                compression,
                                index_levels,
                                block_size,
                                index_key_interval,
                            });
                        }
                    }
                }
            }

            println!("generating the test files...");
            let pb = ProgressBar::new(parameters.len() as u64)
                .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));
            let params_files = parameters
                .into_par_iter()
                .map(|params| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    random_generate_from_params(&mut rng, &folder, &words, &params)
                        .map(|file| (params, file))
                })
                .progress_with(pb)
                .collect::<anyhow::Result<Vec<_>, _>>()?;

            println!("evaluating the test files...");
            let pb = ProgressBar::new(params_files.len() as u64)
                .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));
            let mut results = params_files
                .into_par_iter()
                .map(|(params, mut file)| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let (iter_elapsed, jump_elapsed) = match read_method.as_str() {
                        "direct" => test_cursor(&mut rng, file, &words, entry_count)?,
                        "read-to-vec" => {
                            let mut bytes = Vec::new();
                            file.read_to_end(&mut bytes)?;
                            test_cursor(&mut rng, Cursor::new(bytes), &words, entry_count)?
                        }
                        "bufreader" => {
                            test_cursor(&mut rng, BufReader::new(file), &words, entry_count)?
                        }
                        "memory-mapped" => {
                            let map = unsafe { memmap2::Mmap::map(&file)? };
                            test_cursor(&mut rng, Cursor::new(map), &words, entry_count)?
                        }
                        "memory-mapped-bufreader" => {
                            let map = unsafe { memmap2::Mmap::map(&file)? };
                            test_cursor(
                                &mut rng,
                                BufReader::new(Cursor::new(map)),
                                &words,
                                entry_count,
                            )?
                        }
                        _ => unreachable!(),
                    };

                    Ok((params, iter_elapsed, jump_elapsed))
                })
                .progress_with(pb)
                .collect::<anyhow::Result<Vec<_>>>()?;

            match sort_by.as_str() {
                "iter-only" => results.sort_unstable_by_key(|(_, iter, _)| *iter),
                "iter-and-jump" => results.sort_unstable_by_key(|(_, iter, jump)| *iter + *jump),
                "jump-only" => results.sort_unstable_by_key(|(_, _, jump)| *jump),
                _ => unreachable!(),
            }

            for (params, iter_elapsed, jump_elapsed) in results {
                println!("{:#?}", params);
                println!("took {:.02?} to iterate over values", iter_elapsed);
                println!("took {:.02?} to jump over values", jump_elapsed);
                println!();
            }
        }
        SubCommand::ExtendedTests { seed, file, read_method, sort_by } => {
            let file =
                File::open(&file).with_context(|| format!("while opening {}", file.display()))?;
            let map = unsafe { memmap2::Mmap::map(&file)? };
            let map = Cursor::new(&map);
            let mut cursor = Reader::new(map)?.into_cursor()?;

            println!("extracting the words...");
            let number_of_entries = cursor.len();
            let pb = ProgressBar::new(number_of_entries)
                .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));
            let mut words = Vec::with_capacity(number_of_entries as usize);
            while let Some((k, _)) = cursor.move_on_next()? {
                let word = str::from_utf8(k)?.to_owned();
                words.push(word);
                pb.inc(1);
            }
            pb.finish_and_clear();
            println!("{} unique words extracted!", words.len());

            let compressions =
                vec![CompressionType::None, CompressionType::Snappy, CompressionType::Lz4];
            let index_levels = vec![0, 1, 2, 3];
            let block_sizes = vec![8 * 1024, 4 * 1024, 2 * 1024, 1 * 1024, 512];
            let index_key_intervals: Vec<_> =
                vec![32, 24, 16, 12, 8, 4, 2].into_iter().filter_map(NonZeroUsize::new).collect();

            let mut parameters = Vec::new();
            for &compression in &compressions {
                for &index_levels in &index_levels {
                    for &block_size in &block_sizes {
                        for &index_key_interval in &index_key_intervals {
                            parameters.push(Parameters {
                                compression,
                                index_levels,
                                block_size,
                                index_key_interval,
                            });
                        }
                    }
                }
            }

            println!("generating the test files...");
            let pb = ProgressBar::new(parameters.len() as u64)
                .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));
            let params_files = parameters
                .into_par_iter()
                .map_with(cursor, |cursor, params| {
                    generate_from_params(&folder, cursor, &params).map(|file| (params, file))
                })
                .progress_with(pb)
                .collect::<anyhow::Result<Vec<_>, _>>()?;

            println!("evaluating the test files...");
            let pb = ProgressBar::new(params_files.len() as u64)
                .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));
            let mut results = params_files
                .into_par_iter()
                .map(|(params, mut file)| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let (iter_elapsed, jump_elapsed) = match read_method.as_str() {
                        "direct" => test_cursor(&mut rng, file, &words, number_of_entries)?,
                        "read-to-vec" => {
                            let mut bytes = Vec::new();
                            file.read_to_end(&mut bytes)?;
                            test_cursor(&mut rng, Cursor::new(bytes), &words, number_of_entries)?
                        }
                        "bufreader" => {
                            test_cursor(&mut rng, BufReader::new(file), &words, number_of_entries)?
                        }
                        "memory-mapped" => {
                            let map = unsafe { memmap2::Mmap::map(&file)? };
                            test_cursor(&mut rng, Cursor::new(map), &words, number_of_entries)?
                        }
                        "memory-mapped-bufreader" => {
                            let map = unsafe { memmap2::Mmap::map(&file)? };
                            test_cursor(
                                &mut rng,
                                BufReader::new(Cursor::new(map)),
                                &words,
                                number_of_entries,
                            )?
                        }
                        _ => unreachable!(),
                    };

                    Ok((params, iter_elapsed, jump_elapsed))
                })
                .progress_with(pb)
                .collect::<anyhow::Result<Vec<_>>>()?;

            match sort_by.as_str() {
                "iter-only" => results.sort_unstable_by_key(|(_, iter, _)| *iter),
                "iter-and-jump" => results.sort_unstable_by_key(|(_, iter, jump)| *iter + *jump),
                "jump-only" => results.sort_unstable_by_key(|(_, _, jump)| *jump),
                _ => unreachable!(),
            }

            for (params, iter_elapsed, jump_elapsed) in results {
                println!("{:#?}", params);
                println!("took {:.02?} to iterate over values", iter_elapsed);
                println!("took {:.02?} to jump over values", jump_elapsed);
                println!();
            }
        }
        SubCommand::OneTest {
            seed,
            file,
            read_method,
            compression,
            index_levels,
            block_size,
            index_key_interval,
        } => {
            let file =
                File::open(&file).with_context(|| format!("while opening {}", file.display()))?;
            let map = unsafe { memmap2::Mmap::map(&file)? };
            let map = Cursor::new(&map);
            let mut cursor = Reader::new(map)?.into_cursor()?;

            println!("extracting the words...");
            let number_of_entries = cursor.len();
            let pb = ProgressBar::new(number_of_entries)
                .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));
            let mut words = Vec::with_capacity(number_of_entries as usize);
            while let Some((k, _)) = cursor.move_on_next()? {
                let word = str::from_utf8(k)?.to_owned();
                words.push(word);
                pb.inc(1);
            }
            pb.finish_and_clear();
            println!("{} unique words extracted!", words.len());

            let params = Parameters {
                compression: compression.unwrap_or_default(),
                index_levels,
                block_size,
                index_key_interval,
            };
            let mut file = generate_from_params(&folder, &mut cursor, &params)?;

            let mut rng = StdRng::seed_from_u64(seed);
            let (iter_elapsed, jump_elapsed) = match read_method.as_str() {
                "direct" => test_cursor(&mut rng, file, &words, number_of_entries)?,
                "read-to-vec" => {
                    let mut bytes = Vec::new();
                    file.read_to_end(&mut bytes)?;
                    test_cursor(&mut rng, Cursor::new(bytes), &words, number_of_entries)?
                }
                "bufreader" => {
                    test_cursor(&mut rng, BufReader::new(file), &words, number_of_entries)?
                }
                "memory-mapped" => {
                    let map = unsafe { memmap2::Mmap::map(&file)? };
                    test_cursor(&mut rng, Cursor::new(map), &words, number_of_entries)?
                }
                "memory-mapped-bufreader" => {
                    let map = unsafe { memmap2::Mmap::map(&file)? };
                    test_cursor(
                        &mut rng,
                        BufReader::new(Cursor::new(map)),
                        &words,
                        number_of_entries,
                    )?
                }
                _ => unreachable!(),
            };

            println!("{:#?}", params);
            println!("took {:.02?} to iterate over values", iter_elapsed);
            println!("took {:.02?} to jump over values", jump_elapsed);
            println!();
        }
        SubCommand::OneRandomTest {
            seed,
            entry_count,
            read_method,
            compression,
            index_levels,
            block_size,
            index_key_interval,
        } => {
            println!("generating random words...");
            let mut rng = StdRng::seed_from_u64(seed);
            let pb = ProgressBar::new(entry_count)
                .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));
            let mut words: Vec<_> = iter::repeat_with(|| {
                Gabble::new().with_length(rng.gen_range(3..=15)).generate(&mut rng)
            })
            .take(entry_count as usize)
            .progress_with(pb)
            .collect();
            words.sort_unstable();
            words.dedup();
            println!("{} unique words generated!", words.len());

            let params = Parameters {
                compression: compression.unwrap_or_default(),
                index_levels,
                block_size,
                index_key_interval,
            };
            let mut file = random_generate_from_params(&mut rng, &folder, &words, &params)?;

            let (iter_elapsed, jump_elapsed) = match read_method.as_str() {
                "direct" => test_cursor(&mut rng, file, &words, entry_count)?,
                "read-to-vec" => {
                    let mut bytes = Vec::new();
                    file.read_to_end(&mut bytes)?;
                    test_cursor(&mut rng, Cursor::new(bytes), &words, entry_count)?
                }
                "bufreader" => test_cursor(&mut rng, BufReader::new(file), &words, entry_count)?,
                "memory-mapped" => {
                    let map = unsafe { memmap2::Mmap::map(&file)? };
                    test_cursor(&mut rng, Cursor::new(map), &words, entry_count)?
                }
                "memory-mapped-bufreader" => {
                    let map = unsafe { memmap2::Mmap::map(&file)? };
                    test_cursor(&mut rng, BufReader::new(Cursor::new(map)), &words, entry_count)?
                }
                _ => unreachable!(),
            };

            println!("{:#?}", params);
            println!("took {:.02?} to iterate over values", iter_elapsed);
            println!("took {:.02?} to jump over values", jump_elapsed);
            println!();
        }
        SubCommand::OneLmdbTest { seed, file } => {
            let file =
                File::open(&file).with_context(|| format!("while opening {}", file.display()))?;
            let map = unsafe { memmap2::Mmap::map(&file)? };
            let map = Cursor::new(&map);
            let mut cursor = Reader::new(map)?.into_cursor()?;

            println!("extracting the words...");
            let number_of_entries = cursor.len();
            let pb = ProgressBar::new(number_of_entries)
                .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));
            let mut words = Vec::with_capacity(number_of_entries as usize);
            while let Some((k, _)) = cursor.move_on_next()? {
                let word = str::from_utf8(k)?.to_owned();
                words.push(word);
                pb.inc(1);
            }
            pb.finish_and_clear();
            println!("{} unique words extracted!", words.len());

            let env = generate_lmdb(&folder, &mut cursor)?;
            let database = env.open_database(None)?.unwrap();
            let rtxn = env.read_txn()?;

            let mut rng = StdRng::seed_from_u64(seed);
            let (iter_elapsed, jump_elapsed) =
                test_lmdb(&mut rng, &rtxn, database, &words, number_of_entries)?;

            println!("took {:.02?} to iterate over values", iter_elapsed);
            println!("took {:.02?} to jump over values", jump_elapsed);
            println!();
        }
        SubCommand::OneRandomLmdbTest { seed, entry_count } => {
            println!("generating random words...");
            let mut rng = StdRng::seed_from_u64(seed);
            let pb = ProgressBar::new(entry_count)
                .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));
            let mut words: Vec<_> = iter::repeat_with(|| {
                Gabble::new().with_length(rng.gen_range(3..=15)).generate(&mut rng)
            })
            .take(entry_count as usize)
            .progress_with(pb)
            .collect();
            words.sort_unstable();
            words.dedup();
            println!("{} unique words generated!", words.len());

            let env = random_generate_lmdb(&mut rng, &folder, &words)?;
            let database = env.open_database(None)?.unwrap();
            let rtxn = env.read_txn()?;

            let (iter_elapsed, jump_elapsed) =
                test_lmdb(&mut rng, &rtxn, database, &words, entry_count)?;

            println!("took {:.02?} to iterate over values", iter_elapsed);
            println!("took {:.02?} to jump over values", jump_elapsed);
            println!();
        }
    }

    Ok(())
}

fn test_cursor<RN: Rng, R: io::Read + io::Seek>(
    mut rng: RN,
    reader: R,
    words: &[String],
    entry_count: u64,
) -> anyhow::Result<(Duration, Duration)> {
    let mut cursor = Reader::new(reader)?.into_cursor()?;

    let before_iter = Instant::now();
    let mut i = 0;
    while let Some((k, v)) = cursor.move_on_next()? {
        assert_eq!(k, words[i].as_bytes());
        let bitmap = RoaringBitmap::deserialize_from(v).unwrap();
        assert!(bitmap.len() <= MAX_BITMAP_LEN as u64);
        i += 1;
    }
    let iter_elapsed = before_iter.elapsed();

    let before_jump = Instant::now();
    for _ in 0..entry_count {
        let word = words.choose(&mut rng).unwrap();
        let (k, v) = cursor.move_on_key_greater_than_or_equal_to(&word)?.unwrap();
        assert_eq!(k, word.as_bytes());
        let bitmap = RoaringBitmap::deserialize_from(v).unwrap();
        assert!(bitmap.len() <= MAX_BITMAP_LEN as u64);
        i += 1;
    }

    Ok((iter_elapsed, before_jump.elapsed()))
}

fn test_lmdb<RN: Rng>(
    mut rng: RN,
    rtxn: &RoTxn,
    database: Database,
    words: &[String],
    entry_count: u64,
) -> anyhow::Result<(Duration, Duration)> {
    let before_iter = Instant::now();
    let mut i = 0;
    for result in database.iter(rtxn)? {
        let (k, v) = result?;
        assert_eq!(k, words[i].as_bytes());
        let bitmap = RoaringBitmap::deserialize_from(v).unwrap();
        assert!(bitmap.len() <= MAX_BITMAP_LEN as u64);
        i += 1;
    }
    let iter_elapsed = before_iter.elapsed();

    let before_jump = Instant::now();
    for _ in 0..entry_count {
        let word = words.choose(&mut rng).unwrap();
        let (k, v) = database.get_greater_than_or_equal_to(rtxn, &word)?.unwrap();
        assert_eq!(k, word.as_bytes());
        let bitmap = RoaringBitmap::deserialize_from(v).unwrap();
        assert!(bitmap.len() <= MAX_BITMAP_LEN as u64);
        i += 1;
    }

    Ok((iter_elapsed, before_jump.elapsed()))
}

fn name_from_params(params: &Parameters) -> String {
    format!(
        "{:?}.{}.{}.{}.grd",
        params.compression, params.index_levels, params.block_size, params.index_key_interval
    )
}

fn random_generate_from_params<P: AsRef<Path>, R: Rng>(
    mut rng: R,
    folder: P,
    words: &[String],
    params: &Parameters,
) -> anyhow::Result<File> {
    let filename = name_from_params(params);
    let filepath = folder.as_ref().join(filename);
    match OpenOptions::new().create_new(true).write(true).read(true).open(&filepath) {
        Ok(file) => {
            let mut writer = WriterBuilder::new()
                .compression_type(params.compression)
                .index_levels(params.index_levels)
                .block_size(params.block_size)
                .index_key_interval(params.index_key_interval)
                .build(BufWriter::new(file));

            let mut buffer = Vec::new();

            for word in words {
                random_generate_roaring(&mut rng, &mut buffer);
                writer.insert(word, &buffer)?;
            }

            Ok(writer.into_inner()?.into_inner()?)
        }
        Err(e) if e.kind() == ErrorKind::AlreadyExists => Ok(File::open(filepath)?),
        Err(e) => Err(e.into()),
    }
}

fn random_generate_lmdb<P: AsRef<Path>, R: Rng>(
    mut rng: R,
    folder: P,
    words: &[String],
) -> anyhow::Result<Env> {
    let filepath = folder.as_ref().join("random-lmdb").with_extension("mdb");
    fs::create_dir_all(&filepath)?;

    let env = EnvOpenOptions::new().map_size(FIVE_GIB).open(filepath)?;
    let database = env.create_database(None)?;
    let mut wtxn = env.write_txn()?;

    let mut buffer = Vec::new();

    println!("Inserting values in LMDB...");
    let pb = ProgressBar::new(words.len() as u64)
        .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));
    for word in words.into_iter().progress_with(pb) {
        random_generate_roaring(&mut rng, &mut buffer);
        database.append(&mut wtxn, word, &buffer)?;
    }

    wtxn.commit()?;

    Ok(env)
}

fn random_generate_roaring<R: Rng>(rng: &mut R, buffer: &mut Vec<u8>) {
    buffer.clear();
    let start: u32 = rng.gen();
    let end: u32 = start.saturating_add(rng.gen());
    let roaring =
        RoaringBitmap::from_sorted_iter((start..=end).filter(|_| rng.gen()).take(MAX_BITMAP_LEN))
            .unwrap();
    roaring.serialize_into(buffer).unwrap();
}

fn generate_from_params<P: AsRef<Path>, R: io::Read + io::Seek>(
    folder: P,
    cursor: &mut ReaderCursor<R>,
    params: &Parameters,
) -> anyhow::Result<File> {
    let filename = name_from_params(params);
    let filepath = folder.as_ref().join(filename);
    match OpenOptions::new().create_new(true).write(true).read(true).open(&filepath) {
        Ok(file) => {
            let mut writer = WriterBuilder::new()
                .compression_type(params.compression)
                .index_levels(params.index_levels)
                .block_size(params.block_size)
                .index_key_interval(params.index_key_interval)
                .build(BufWriter::new(file));

            cursor.reset();
            while let Some((k, v)) = cursor.move_on_next()? {
                writer.insert(k, v)?;
            }

            Ok(writer.into_inner()?.into_inner()?)
        }
        Err(e) if e.kind() == ErrorKind::AlreadyExists => Ok(File::open(filepath)?),
        Err(e) => Err(e.into()),
    }
}

fn generate_lmdb<P: AsRef<Path>, R: io::Read + io::Seek>(
    folder: P,
    cursor: &mut ReaderCursor<R>,
) -> anyhow::Result<Env> {
    let filepath = folder.as_ref().join("lmdb").with_extension("mdb");
    fs::create_dir_all(&filepath)?;

    let env = EnvOpenOptions::new().map_size(FIVE_GIB).open(filepath)?;
    let database = env.create_database(None)?;
    let mut wtxn = env.write_txn()?;

    println!("Inserting values in LMDB...");
    let pb = ProgressBar::new(cursor.len())
        .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));

    cursor.reset();
    while let Some((k, v)) = cursor.move_on_next()? {
        database.append(&mut wtxn, k, v)?;
        pb.inc(1);
    }

    pb.finish_and_clear();

    wtxn.commit()?;

    Ok(env)
}
