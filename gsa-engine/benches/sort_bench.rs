//! Compares GSA against Rust's built-in `sort_unstable` across a range of
//! input sizes. Run with `cargo build --release --bench sort_bench` then
//! execute the produced binary (this crate uses a hand-rolled harness, not
//! `criterion`, to keep dependencies minimal).
//!
//! Methodology, for reproducibility:
//! - Input: uniformly random `f32` in `[-1e9, 1e9)`, generated with a
//!   fixed PRNG seed (`SEED` below) so every run sees identical input.
//! - Each size is measured across `ITERATIONS` runs on freshly-cloned,
//!   freshly-shuffled-order data (the same seeded array, re-cloned each
//!   iteration so neither sort ever benefits from the other's now-sorted
//!   output); the first iteration is a discarded warm-up (page faults,
//!   allocator/rayon-pool warm-up) and not counted in the reported stats.
//! - Reported: median, min, and max across the counted iterations, not a
//!   single sample — wall-clock timing on a shared, unpinned machine is
//!   noisy enough that a single run isn't representative.
//! - `elapsed_ms` for GSA is measured the same way the production
//!   WebSocket server measures it (`SortRunStats::elapsed`), i.e. it's the
//!   real code path, not a proxy.

use gsa_engine::gpu::GpuContext;
use gsa_engine::sort::gsa_sort;
use gsa_engine::threadpool::build_pool;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::mpsc::channel;
use std::time::Instant;

const SEED: u64 = 42;
const ITERATIONS: usize = 7; // +1 discarded warm-up iteration per size

fn random_input(n: usize, rng: &mut StdRng) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(-1e9..1e9)).collect()
}

fn bench_gsa(data: &mut [f32], pool: &rayon::ThreadPool, gpu: Option<&GpuContext>) -> f64 {
    let (tx, rx) = channel();
    let start = Instant::now();
    gsa_sort(data, pool, gpu, &tx);
    while let Ok(event) = rx.recv() {
        if let gsa_engine::sort::SortEvent::Done(_) = event {
            break;
        }
    }
    start.elapsed().as_secs_f64() * 1000.0
}

fn bench_builtin(data: &mut [f32]) -> f64 {
    let start = Instant::now();
    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    start.elapsed().as_secs_f64() * 1000.0
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    let mid = xs.len() / 2;
    if xs.len() % 2 == 0 {
        (xs[mid - 1] + xs[mid]) / 2.0
    } else {
        xs[mid]
    }
}

fn main() {
    println!("=== GSA vs sort_unstable benchmark ===");
    let pool_info = build_pool();
    let gpu = GpuContext::new();
    match &gpu {
        Some(ctx) => println!("GPU available: {}", ctx.device_name),
        None => println!("GPU not available: CPU-only fallback path will be exercised"),
    }
    println!(
        "CPU pool: {} / {} logical cores",
        pool_info.threads_claimed, pool_info.logical_cores
    );
    println!(
        "seed={SEED}, iterations={ITERATIONS} (+1 discarded warm-up), dtype=f32, distribution=uniform[-1e9,1e9)\n"
    );

    let sizes = [1_000usize, 10_000, 100_000, 1_000_000, 5_000_000];

    println!(
        "{:>10} | {:>27} | {:>27} | {:>8}",
        "n", "gsa median (min-max) ms", "sort_unstable median (min-max) ms", "speedup"
    );
    println!("{}", "-".repeat(84));

    for &n in &sizes {
        let mut rng = StdRng::seed_from_u64(SEED ^ (n as u64));
        let base = random_input(n, &mut rng);

        // Discarded warm-up iteration.
        {
            let mut d = base.clone();
            bench_gsa(&mut d, &pool_info.pool, gpu.as_ref());
            let mut d = base.clone();
            bench_builtin(&mut d);
        }

        let mut gsa_times = Vec::with_capacity(ITERATIONS);
        let mut builtin_times = Vec::with_capacity(ITERATIONS);
        for _ in 0..ITERATIONS {
            let mut gsa_data = base.clone();
            let gsa_ms = bench_gsa(&mut gsa_data, &pool_info.pool, gpu.as_ref());
            assert!(
                gsa_data.windows(2).all(|w| w[0] <= w[1]),
                "GSA output not sorted for n={n}"
            );
            gsa_times.push(gsa_ms);

            let mut builtin_data = base.clone();
            builtin_times.push(bench_builtin(&mut builtin_data));
        }

        let gsa_median = median(gsa_times.clone());
        let gsa_min = gsa_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let gsa_max = gsa_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let builtin_median = median(builtin_times.clone());
        let builtin_min = builtin_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let builtin_max = builtin_times
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        println!(
            "{:>10} | {:>8.3} ({:>6.3}-{:>6.3}) | {:>8.3} ({:>6.3}-{:>6.3}) | {:>7.2}x",
            n,
            gsa_median,
            gsa_min,
            gsa_max,
            builtin_median,
            builtin_min,
            builtin_max,
            builtin_median / gsa_median.max(0.0001)
        );
    }
}
