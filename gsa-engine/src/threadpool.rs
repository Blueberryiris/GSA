//! Thread pool sizing.
//!
//! GSA deliberately claims at least 70% of logical cores for its rayon
//! pool, rounded up, so the partition and merge phases visibly saturate
//! the machine rather than leaving cores idle "for safety."

pub struct PoolInfo {
    pub pool: rayon::ThreadPool,
    pub logical_cores: usize,
    pub threads_claimed: usize,
}

pub fn build_pool() -> PoolInfo {
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let threads_claimed = ((logical_cores as f64) * 0.70).ceil() as usize;
    let threads_claimed = threads_claimed.max(1);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads_claimed)
        .thread_name(|i| format!("gsa-worker-{i}"))
        .build()
        .expect("failed to build GSA thread pool");

    PoolInfo {
        pool,
        logical_cores,
        threads_claimed,
    }
}
