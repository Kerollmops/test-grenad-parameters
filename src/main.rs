use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Cursor, ErrorKind, Read};
use std::iter;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use gabble::Gabble;
use grenad::{CompressionType, Reader, WriterBuilder};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use roaring::RoaringBitmap;

const MAX_BITMAP_LEN: usize = 10_000;

#[derive(Parser)]
#[clap(version = "1.0", author = "Kevin K. <kbknapp@gmail.com>")]
struct Opts {
    /// A level of verbosity, and can be used multiple times
    #[clap(short, long, parse(from_occurrences))]
    verbose: i32,

    /// The folder where to create the grenad databases.
    #[clap(long)]
    folder: PathBuf,

    /// The seed to use
    #[clap(long, default_value = "42")]
    seed: u64,

    #[clap(long, default_value = "10000")]
    entry_count: u64,

    #[clap(
        long,
        default_value = "direct",
        possible_values = &["direct", "read-to-vec", "bufreader", "memory-mapped", "memory-mapped-bufreader"],
    )]
    read_method: String,
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
    let Opts { verbose, folder, seed, entry_count, read_method } = Opts::try_parse()?;

    println!("generating random words...");
    let mut rng = StdRng::seed_from_u64(seed);
    let pb = ProgressBar::new(entry_count)
        .with_style(ProgressStyle::default_bar().template("{wide_bar} {pos}/{len} {eta}"));
    let mut words: Vec<_> =
        iter::repeat_with(|| Gabble::new().with_length(rng.gen_range(3..=15)).generate(&mut rng))
            .take(entry_count as usize)
            .progress_with(pb)
            .collect();
    words.sort_unstable();
    words.dedup();
    println!("{} unique words generated!", words.len());

    let compressions = vec![CompressionType::None, CompressionType::Snappy, CompressionType::Lz4];
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
            generate_from_params(&mut rng, &folder, &words, &params).map(|file| (params, file))
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

            Ok((params, iter_elapsed, jump_elapsed))
        })
        .progress_with(pb)
        .collect::<anyhow::Result<Vec<_>>>()?;

    results.sort_unstable_by_key(|(_, iter_elapsed, jump_elapsed)| {
        /* *iter_elapsed + */
        *jump_elapsed
    });

    for (params, iter_elapsed, jump_elapsed) in results {
        println!("{:#?}", params);
        println!("took {:.02?} to iterate over values", iter_elapsed);
        println!("took {:.02?} to jump over values", jump_elapsed);
        println!();
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
    while let Some((k, _v)) = cursor.move_on_next()? {
        assert_eq!(k, words[i].as_bytes());
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

fn name_from_params(params: &Parameters) -> String {
    format!(
        "{:?}.{}.{}.{}.grd",
        params.compression, params.index_levels, params.block_size, params.index_key_interval
    )
}

fn generate_from_params<P: AsRef<Path>, R: Rng>(
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
                generate_roaring(&mut rng, &mut buffer);
                writer.insert(word, &buffer)?;
            }

            Ok(writer.into_inner()?.into_inner()?)
        }
        Err(e) if e.kind() == ErrorKind::AlreadyExists => Ok(File::open(filepath)?),
        Err(e) => Err(e.into()),
    }
}

fn generate_roaring<R: Rng>(rng: &mut R, buffer: &mut Vec<u8>) {
    buffer.clear();
    let start: u32 = rng.gen();
    let end: u32 = start.saturating_add(rng.gen());
    let roaring =
        RoaringBitmap::from_sorted_iter((start..=end).filter(|_| rng.gen()).take(MAX_BITMAP_LEN))
            .unwrap();
    roaring.serialize_into(buffer).unwrap();
}
