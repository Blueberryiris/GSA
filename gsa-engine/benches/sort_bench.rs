//! Compares GSA against Rust's built-in `sort_unstable` across a range of
//! input sizes. Run with `cargo run --release --bin sort_bench` semantics
//! via `cargo bench` (this crate uses a hand-rolled harness, not
//! `criterion`, to keep dependencies minimal).

use gsa_engine::gpu::GpuContext;
use gsa_engine::sort::gsa_sort;
use gsa_engine::threadpool::build_pool;
use rand::Rng;
use std::sync::mpsc::channel;
use std::time::Instant;

fn random_input(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
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

fn main() {
    println!("=== GSA vs sort_unstable benchmark ===");
    let pool_info = build_pool();
    let gpu = GpuContext::new();
    match &gpu {
        Some(ctx) => println!("GPU available: {}", ctx.device_name),
        None => println!("GPU not available: CPU-only fallback path will be exercised"),
    }
    println!(
        "CPU pool: {} / {} logical cores\n",
        pool_info.threads_claimed, pool_info.logical_cores
    );

    let sizes = [1_000usize, 10_000, 100_000, 1_000_000, 5_000_000];

    println!(
        "{:>10} | {:>14} | {:>14} | {:>8}",
        "n", "gsa (ms)", "sort_unstable (ms)", "speedup"
    );
    println!("{}", "-".repeat(56));

    for &n in &sizes {
        let base = random_input(n);

        let mut gsa_data = base.clone();
        let gsa_ms = bench_gsa(&mut gsa_data, &pool_info.pool, gpu.as_ref());
        assert!(gsa_data.windows(2).all(|w| w[0] <= w[1]), "GSA output not sorted for n={n}");

        let mut builtin_data = base.clone();
        let builtin_ms = bench_builtin(&mut builtin_data);

        println!(
            "{:>10} | {:>14.3} | {:>14.3} | {:>7.2}x",
            n,
            gsa_ms,
            builtin_ms,
            builtin_ms / gsa_ms.max(0.0001)
        );
    }
}
