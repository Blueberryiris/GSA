use gsa_engine::allocator::ScratchArena;
use gsa_engine::autotune;
use gsa_engine::gpu::GpuContext;
use gsa_engine::server::{self, AppState, DEFAULT_PORT};
use gsa_engine::sort::DEFAULT_BUCKET_MULTIPLIER;
use gsa_engine::threadpool::build_pool;
use std::sync::Arc;

/// Fixed showcase allocation: GSA always reserves ~4 GB of RAM on startup
/// regardless of the size of the array it ends up sorting. See
/// `allocator.rs` for why this is intentional.
const SCRATCH_ARENA_BYTES: usize = 4 * 1024 * 1024 * 1024;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    println!("=== GSA Engine starting ===");

    // --- Memory ---
    print!(
        "[gsa-engine] reserving {:.1} GB scratch arena... ",
        SCRATCH_ARENA_BYTES as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let arena = Arc::new(ScratchArena::new(SCRATCH_ARENA_BYTES));
    println!(
        "committed {:.2} GB",
        arena.committed_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    // The arena itself isn't threaded through the sort path in this
    // showcase build (buckets are sized dynamically per run), but a
    // background thread keeps re-touching it for the lifetime of the
    // process so macOS's memory compressor can't quietly shrink the
    // reservation while the engine is idle between sort requests.
    arena.spawn_keepalive();

    // --- CPU ---
    let pool_info = build_pool();
    println!(
        "[gsa-engine] CPU thread pool: {} / {} logical cores claimed ({:.0}%)",
        pool_info.threads_claimed,
        pool_info.logical_cores,
        100.0 * pool_info.threads_claimed as f64 / pool_info.logical_cores as f64
    );

    // --- GPU ---
    let gpu = GpuContext::new();
    match &gpu {
        Some(ctx) => {
            println!(
                "[gsa-engine] GPU: {} (max {} threads/threadgroup, execution width {})",
                ctx.device_name, ctx.max_threads_per_threadgroup, ctx.thread_execution_width
            );
        }
        None => {
            println!("[gsa-engine] no Metal-compatible GPU found — using CPU radix sort for every request");
        }
    }

    // --- Autotune ---
    // None of GSA's sorting primitives are novel (sample sort, bitonic
    // sort, radix sort are all textbook algorithms — see each module's
    // doc comment). What isn't off-the-shelf: GSA measures its own real
    // sort code against a calibration array at startup to decide (a)
    // whether the GPU path is actually faster than CPU radix sort on
    // *this* machine at all, and (b), if so, what bucket granularity to
    // use — instead of shipping fixed constants for every Mac.
    let bucket_multiplier = if let Some(gpu_ctx) = &gpu {
        print!("[gsa-engine] autotuning: GPU bitonic vs CPU radix on this machine... ");
        let result = autotune::calibrate(&pool_info.pool, gpu_ctx);
        println!("done");
        for trial in &result.trials {
            let is_best = (trial.elapsed_ms - result.gpu_best_ms).abs() < f64::EPSILON;
            println!(
                "[gsa-engine]   GPU  {:>4.1}x threads: {:>8.3} ms{}",
                trial.multiplier,
                trial.elapsed_ms,
                if is_best { " <- best GPU config" } else { "" }
            );
        }
        println!(
            "[gsa-engine]   CPU  radix sort:     {:>8.3} ms",
            result.radix_ms
        );
        if result.gpu_selected() {
            println!(
                "[gsa-engine] GPU wins on this machine ({:.3} ms vs radix's {:.3} ms) — using {:.1}x threads as the bucket multiplier for arrays >= {} elements",
                result.gpu_best_ms, result.radix_ms, result.bucket_multiplier, gsa_engine::sort::GPU_THRESHOLD
            );
        } else {
            println!(
                "[gsa-engine] CPU radix sort wins on this machine ({:.3} ms vs GPU's {:.3} ms) — GPU path disabled, every request uses radix sort",
                result.radix_ms, result.gpu_best_ms
            );
        }
        result.bucket_multiplier
    } else {
        DEFAULT_BUCKET_MULTIPLIER
    };

    let state = Arc::new(AppState {
        pool: pool_info.pool,
        gpu,
        bucket_multiplier,
    });

    // --- Network ---
    let port: u16 = std::env::var("GSA_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    let lan_ip = server::discover_lan_ip();
    println!("[gsa-engine] listening on 0.0.0.0:{port}");
    println!("[gsa-engine] connect from any device on this network at: ws://{lan_ip}:{port}");
    println!(
        "[gsa-engine] (macOS may prompt to allow incoming connections on first run — accept it)"
    );
    println!("=== GSA Engine ready ===");

    server::run(state, port).await;
}
