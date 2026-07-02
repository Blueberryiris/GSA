//! Startup self-calibration.
//!
//! None of GSA's individual sorting primitives are novel — sample sort,
//! bitonic sort, and radix sort are all decades-old, well-documented
//! algorithms (see `sort.rs`, `gpu.rs`, `radix.rs` for citations). What
//! most sorting libraries *don't* do is measure the machine they're
//! actually running on and adapt to it. Calibration answers two questions
//! GSA can't answer correctly from a fixed constant:
//!
//! 1. **Does the GPU path even win on this machine?** A sweep across this
//!    project's own dev hardware (500 to 40,000,000 elements) found GSA's
//!    parallel CPU radix sort beating the GPU-dispatched bitonic sort at
//!    *every* size tested, by 2x-11x — bitonic sort's O(n log^2 n)
//!    compare-exchange count loses to radix sort's O(n) once there's
//!    enough data for the complexity gap to matter, and GPU dispatch/
//!    upload overhead makes it worse for small n too. But that's one
//!    measurement on one machine; a Mac with a relatively weaker CPU and
//!    stronger GPU (or vice versa) could have a different answer. Rather
//!    than bake in "GPU never wins" as a hardcoded assumption, GSA times
//!    both strategies on a calibration array at startup and only takes
//!    the GPU path if it actually measures faster on *this* hardware.
//! 2. **If the GPU path does win, how many sample-sort buckets should it
//!    use?** Too few and the CPU partition/merge phases don't have enough
//!    independent buckets to spread across the thread pool; too many and
//!    per-bucket fixed costs (padding, buffer upload, dispatch setup)
//!    start to dominate. The right answer depends on this machine's GPU
//!    dispatch latency versus its CPU thread-switch/partition cost.
//!
//! Both are answered by running GSA's *actual* sort code
//! (`sort::gsa_sort_tuned`, not a synthetic proxy) against a calibration
//! array once at startup and keeping whatever wins for the life of the
//! process. This is the same idea autotuning libraries like FFTW and
//! ATLAS/OpenBLAS use for their own kernels.

use crate::gpu::GpuContext;
use crate::radix::radix_sort_f32_parallel;
use crate::sort::{gsa_sort_tuned, NullSender};
use rand::Rng;
use std::time::Instant;

/// Size of the synthetic array used for calibration trials. Large enough
/// that GPU dispatch overhead, radix sort's per-pass cost, and
/// partition/merge costs all show up clearly in the timing; small enough
/// that the whole sweep finishes in well under a second.
const CALIBRATION_SIZE: usize = 400_000;

/// Bucket-count multipliers tried during calibration (as a multiple of
/// the CPU thread pool's size).
const CANDIDATE_MULTIPLIERS: &[f64] = &[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0];

pub struct Trial {
    pub multiplier: f64,
    pub elapsed_ms: f64,
}

pub struct AutotuneResult {
    /// `0.0` if radix sort won the calibration comparison (GPU path
    /// disabled for this run); otherwise the fastest GPU bucket
    /// multiplier found.
    pub bucket_multiplier: f64,
    pub trials: Vec<Trial>,
    pub radix_ms: f64,
    pub gpu_best_ms: f64,
}

impl AutotuneResult {
    pub fn gpu_selected(&self) -> bool {
        self.bucket_multiplier > 0.0
    }
}

/// Sweep [`CANDIDATE_MULTIPLIERS`] against a calibration array using the
/// real GSA GPU sort path, then compare the best GPU result against GSA's
/// parallel radix sort on the same data, and return whichever strategy
/// (and, if GPU, whichever multiplier) actually won on this machine.
/// Blocking; takes on the order of tens to a few hundred milliseconds
/// depending on hardware, which is why it's run once at startup rather
/// than per-request.
pub fn calibrate(pool: &rayon::ThreadPool, gpu: &GpuContext) -> AutotuneResult {
    let base: Vec<f32> = {
        let mut rng = rand::thread_rng();
        (0..CALIBRATION_SIZE)
            .map(|_| rng.gen_range(-1e6..1e6))
            .collect()
    };

    // Untimed warm-up dispatch: Metal compiles/caches its pipeline state
    // lazily on first use, so whichever candidate ran first would
    // otherwise eat a one-time JIT cost that has nothing to do with its
    // actual bucket multiplier (observed in practice: the first trial can
    // run 40-50x slower than every trial after it). Running one throwaway
    // sort first means every timed trial below starts from the same warm
    // GPU state.
    {
        let mut warm = base.clone();
        gsa_sort_tuned(&mut warm, pool, Some(gpu), 1.0, &NullSender);
    }

    let mut trials = Vec::with_capacity(CANDIDATE_MULTIPLIERS.len());
    for &multiplier in CANDIDATE_MULTIPLIERS {
        let mut data = base.clone();
        let start = Instant::now();
        gsa_sort_tuned(&mut data, pool, Some(gpu), multiplier, &NullSender);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        trials.push(Trial {
            multiplier,
            elapsed_ms,
        });
    }

    let gpu_best = trials
        .iter()
        .min_by(|a, b| a.elapsed_ms.total_cmp(&b.elapsed_ms))
        .expect("CANDIDATE_MULTIPLIERS is non-empty");
    let gpu_best_multiplier = gpu_best.multiplier;
    let gpu_best_ms = gpu_best.elapsed_ms;

    // Compare against GSA's own radix sort on identical data, using the
    // real code path (same pool, same input) rather than a proxy.
    let radix_ms = {
        let mut data = base.clone();
        let start = Instant::now();
        pool.install(|| radix_sort_f32_parallel(&mut data));
        start.elapsed().as_secs_f64() * 1000.0
    };

    let bucket_multiplier = if radix_ms <= gpu_best_ms {
        0.0
    } else {
        gpu_best_multiplier
    };

    AutotuneResult {
        bucket_multiplier,
        trials,
        radix_ms,
        gpu_best_ms,
    }
}
