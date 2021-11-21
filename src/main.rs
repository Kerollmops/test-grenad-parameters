use std::fs::{File, OpenOptions};
use std::io::{self, ErrorKind};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use grenad::{CompressionType, Reader, WriterBuilder};
use indicatif::ParallelProgressIterator;
use rand::rngs::StdRng;
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
    let Opts { verbose, folder, seed, entry_count } = Opts::try_parse()?;

    let compressions = vec![CompressionType::None, CompressionType::Snappy, CompressionType::Lz4];
    let index_levels = vec![0, 1, 2, 3];
    let block_sizes =
        vec![12 * 1024, 10 * 1024, 8 * 1024, 4 * 1024, 2 * 1024, 1 * 1024, 512, 256, 128];
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
    let params_files = parameters
        .into_par_iter()
        .map(|params| {
            let mut rng = StdRng::seed_from_u64(seed);
            generate_from_params(&mut rng, &folder, &params, entry_count).map(|file| (params, file))
        })
        .progress()
        .collect::<anyhow::Result<Vec<_>, _>>()?;

    println!("evaluating the test files...");
    let mut results = params_files
        .into_par_iter()
        .map(|(params, file)| {
            let mut rng = StdRng::seed_from_u64(seed);
            let (iter_elapsed, jump_elapsed) = test_cursor(&mut rng, file, entry_count)?;
            Ok((params, iter_elapsed, jump_elapsed))
        })
        .progress()
        .collect::<anyhow::Result<Vec<_>>>()?;

    results.sort_unstable_by_key(|(_, _iter_elapsed, jump_elapsed)| *jump_elapsed);

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
    entry_count: u64,
) -> anyhow::Result<(Duration, Duration)> {
    let mut cursor = Reader::new(reader)?.into_cursor()?;

    let before_iter = Instant::now();
    let mut i = 0u64;
    while let Some((k, _v)) = cursor.move_on_next()? {
        assert_eq!(k, i.to_be_bytes());
        i += 1;
    }
    let iter_elapsed = before_iter.elapsed();

    let before_jump = Instant::now();
    for _ in 0..entry_count {
        let n = rng.gen_range(0..entry_count);
        let (k, v) = cursor.move_on_key_greater_than_or_equal_to(n.to_be_bytes())?.unwrap();
        let bitmap = RoaringBitmap::deserialize_from(v).unwrap();
        assert_eq!(k, n.to_be_bytes());
        assert!(bitmap.len() <= MAX_BITMAP_LEN as u64);
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
    params: &Parameters,
    entry_count: u64,
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
                .build(file);

            let mut buffer = Vec::new();

            for i in 0..entry_count {
                let key = i.to_be_bytes();
                generate_roaring(&mut rng, &mut buffer);
                writer.insert(key, &buffer)?;
            }

            Ok(writer.into_inner()?)
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
