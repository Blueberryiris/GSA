//! The GSA hybrid sorting algorithm.
//!
//! GSA picks between three strategies at runtime, in this order:
//!
//! 1. **Direct sort (tiny arrays, below [`DIRECT_SORT_THRESHOLD`]):** just
//!    call `sort_unstable_by` on a single thread. Below a few thousand
//!    elements, every parallel strategy's fixed dispatch/coordination cost
//!    exceeds the actual sorting work, so the fastest thing to do is the
//!    simplest thing — measured on this project's own hardware, GSA's
//!    parallel radix sort was ~20x *slower* than a plain single-threaded
//!    `sort_unstable_by` at n=1000, purely from rayon task-dispatch
//!    overhead on trivial-sized chunks.
//! 2. **Parallel radix sort (`radix.rs`), the default for everything
//!    else.** A comprehensive sweep (500 to 40,000,000 elements) found
//!    this beats GSA's GPU-dispatched bitonic sort at *every single size
//!    tested*, by 2x-11x, with the gap widening as n grows. That's not an
//!    implementation bug — it's algorithmic: bitonic sort is a sorting
//!    network with O(n log^2 n) compare-exchanges, while radix sort is
//!    O(n). No amount of GPU parallelism overcomes that complexity gap
//!    once n is large enough for the difference to matter, and GPU
//!    dispatch/upload overhead makes it worse for small n too.
//! 3. **GPU hybrid sort — sample-sort partition (CPU) / bitonic sort
//!    (GPU) / merge (CPU) — used only if [`autotune::calibrate`] measures
//!    it as *actually faster than radix on this specific machine* at
//!    startup.** Every Mac's CPU/GPU balance differs, so rather than bake
//!    in "GPU never wins" as a hardcoded assumption from one benchmark
//!    run, GSA measures both strategies on the real hardware it's running
//!    on and only takes the GPU path if it wins that measurement. See
//!    `autotune.rs` for the comparison and `gpu.rs` for why the GPU path
//!    itself is as fast as it is now (it used to be dramatically slower).
//!
//! The GPU path, when selected: **partition** (CPU, multithreaded) samples
//! the input, derives pivots, and buckets every element by pivot range
//! (sample sort) in parallel across the rayon pool; **local sort** (GPU)
//! sorts every bucket at or above `GPU_BUCKET_MIN` with a bitonic-sort
//! compute kernel, with all buckets' GPU work batched into one Metal
//! command buffer submitted once; **merge** (CPU, multithreaded) exploits
//! that sample-sort buckets are already disjoint, pivot-ordered value
//! ranges, so writing each sorted bucket into its prefix-sum offset *is*
//! the merge — done as one parallel scatter, each bucket's placement
//! reported as its own progress frame.

use crate::gpu::{next_power_of_two, GpuContext};
use crate::radix::radix_sort_f32_parallel;
use rand::Rng;
use rayon::prelude::*;
use std::time::{Duration, Instant};

/// Arrays smaller than this sort directly on a single thread — see the
/// module doc comment for why parallel strategies lose here.
pub const DIRECT_SORT_THRESHOLD: usize = 15_000;
/// Arrays smaller than this skip the GPU path entirely even if it's the
/// selected strategy (a floor under `autotune`'s decision, not the main
/// gate — see the module doc comment).
pub const GPU_THRESHOLD: usize = 50_000;
/// Buckets smaller than this sort on the CPU; GPU dispatch overhead isn't
/// worth it for a handful of elements.
const GPU_BUCKET_MIN: usize = 256;
/// Bucket count multiplier used before autotuning has run (or if it's
/// skipped, e.g. no GPU present). `0.0` means "don't use the GPU path at
/// all" — the safe default, since GPU only ever wins if `autotune`
/// measures it winning on this specific machine. See `autotune.rs`.
pub const DEFAULT_BUCKET_MULTIPLIER: f64 = 0.0;

#[derive(Debug, Clone)]
pub struct SortRunStats {
    pub elapsed: Duration,
    pub elements: usize,
    pub algorithm: &'static str,
    pub threads_used: usize,
    pub gpu_used: bool,
    pub gpu_device: Option<String>,
    /// The bucket-count multiplier actually used for this run's GPU path
    /// (1.0 unless the engine autotuned a different value at startup).
    /// `None` for runs that didn't take the GPU path at all.
    pub bucket_multiplier: Option<f64>,
}

pub enum SortEvent {
    /// A partial update: the values now at `indices` (same order/length).
    Progress {
        indices: Vec<usize>,
        values: Vec<f32>,
    },
    Done(SortRunStats),
}

/// Abstracts over the channel type used to report progress, so the same
/// sort code can be driven from a plain `std::sync::mpsc` in tests and
/// from a `tokio::sync::mpsc::UnboundedSender` in the WebSocket server
/// (whose `send` is a non-blocking, non-async call safe to invoke from a
/// `spawn_blocking` thread).
pub trait ProgressSender: Send {
    fn send(&self, event: SortEvent);
}

impl ProgressSender for std::sync::mpsc::Sender<SortEvent> {
    fn send(&self, event: SortEvent) {
        let _ = std::sync::mpsc::Sender::send(self, event);
    }
}

impl ProgressSender for tokio::sync::mpsc::UnboundedSender<SortEvent> {
    fn send(&self, event: SortEvent) {
        let _ = tokio::sync::mpsc::UnboundedSender::send(self, event);
    }
}

/// A `ProgressSender` that discards every event. Used by the autotuner,
/// which needs to run the real `gsa_sort_tuned` code path (not a separate
/// proxy benchmark) but doesn't care about progress frames.
pub struct NullSender;
impl ProgressSender for NullSender {
    fn send(&self, _event: SortEvent) {}
}

/// Run GSA on `data` in place, reporting progress frames and a final
/// completion event through `tx`. Intended to be called from a blocking
/// context (e.g. `tokio::task::spawn_blocking`) since it runs synchronously
/// and can take a while for large inputs. Uses [`DEFAULT_BUCKET_MULTIPLIER`]
/// for the GPU path's bucket count; see [`gsa_sort_tuned`] for the
/// autotuned variant the server actually uses once calibration has run.
pub fn gsa_sort(
    data: &mut [f32],
    pool: &rayon::ThreadPool,
    gpu: Option<&GpuContext>,
    tx: &dyn ProgressSender,
) {
    gsa_sort_tuned(data, pool, gpu, DEFAULT_BUCKET_MULTIPLIER, tx)
}

/// Same as [`gsa_sort`], but with the GPU path's bucket count expressed as
/// `threads_used * bucket_multiplier` instead of a fixed 1x. This is the
/// knob GSA's startup autotuning (`autotune.rs`) actually searches over:
/// the best multiplier depends on the real tradeoff between this
/// machine's GPU dispatch latency and its CPU thread-switch/partition
/// cost, which fixed constants can't account for across different Macs.
pub fn gsa_sort_tuned(
    data: &mut [f32],
    pool: &rayon::ThreadPool,
    gpu: Option<&GpuContext>,
    bucket_multiplier: f64,
    tx: &dyn ProgressSender,
) {
    let start = Instant::now();
    let n = data.len();
    let threads_used = pool.current_num_threads();

    // These monolithic (non-bucketed) paths have no meaningful mid-sort
    // state to animate — unlike the GPU path's per-bucket placement, there
    // are no partial results until the whole call returns — and `Done`
    // already carries the complete sorted array. So they skip the
    // Progress frame entirely rather than paying for a redundant full
    // array clone (`data.to_vec()`) purely to send the same data twice
    // over the wire. `elapsed` is also captured immediately after the
    // sort call returns, not after building/sending anything else, so it
    // measures the algorithm itself rather than incidental bookkeeping.

    if n < 2 {
        let elapsed = start.elapsed();
        let _ = tx.send(SortEvent::Done(SortRunStats {
            elapsed,
            elements: n,
            algorithm: "trivial",
            threads_used,
            gpu_used: false,
            gpu_device: None,
            bucket_multiplier: None,
        }));
        return;
    }

    if n < DIRECT_SORT_THRESHOLD {
        data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let elapsed = start.elapsed();
        let _ = tx.send(SortEvent::Done(SortRunStats {
            elapsed,
            elements: n,
            algorithm: "direct-sort-tiny",
            threads_used,
            gpu_used: false,
            gpu_device: None,
            bucket_multiplier: None,
        }));
        return;
    }

    // GPU is only used if `bucket_multiplier > 0.0` — set by `autotune`
    // only when it measures the GPU path as actually faster than radix on
    // this specific machine at startup. See the module doc comment.
    let use_gpu = gpu.is_some() && n >= GPU_THRESHOLD && bucket_multiplier > 0.0;

    if !use_gpu {
        pool.install(|| {
            radix_sort_f32_parallel(data);
        });
        let elapsed = start.elapsed();
        let _ = tx.send(SortEvent::Done(SortRunStats {
            elapsed,
            elements: n,
            algorithm: "cpu-parallel-radix",
            threads_used,
            gpu_used: false,
            gpu_device: None,
            bucket_multiplier: None,
        }));
        return;
    }

    let gpu_ctx = gpu.unwrap();
    let num_buckets = ((threads_used as f64) * bucket_multiplier).round().max(1.0) as usize;

    let mut buckets: Vec<Vec<f32>> = pool.install(|| partition_into_buckets(data, num_buckets));

    // Phase 2: local sort. Small buckets sort on the CPU (GPU dispatch
    // overhead isn't worth it for a handful of values); everything else is
    // padded and handed to the GPU as one batch, so the whole phase costs
    // a single CPU/GPU synchronization instead of one per bucket.
    let mut gpu_indices = Vec::new();
    let mut gpu_padded: Vec<Vec<f32>> = Vec::new();
    for (i, bucket) in buckets.iter_mut().enumerate() {
        if bucket.len() < 2 {
            continue;
        }
        if bucket.len() < GPU_BUCKET_MIN {
            bucket.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        } else {
            let padded_len = next_power_of_two(bucket.len());
            let mut padded = vec![f32::INFINITY; padded_len];
            padded[..bucket.len()].copy_from_slice(bucket);
            gpu_indices.push(i);
            gpu_padded.push(padded);
        }
    }
    if !gpu_padded.is_empty() {
        gpu_ctx.sort_padded_buckets(&mut gpu_padded);
        for (padded, bucket_i) in gpu_padded.into_iter().zip(gpu_indices) {
            let len = buckets[bucket_i].len();
            buckets[bucket_i].copy_from_slice(&padded[..len]);
        }
    }

    // Phase 3: merge (parallel scatter of pivot-ordered buckets).
    let lens: Vec<usize> = buckets.iter().map(|b| b.len()).collect();
    let mut offsets = Vec::with_capacity(lens.len());
    let mut acc = 0usize;
    for &l in &lens {
        offsets.push(acc);
        acc += l;
    }

    // The actual copy runs in parallel across the CPU pool; progress
    // events are collected per-bucket and then flushed sequentially from
    // this thread afterward, since not every channel type used here
    // (e.g. `std::sync::mpsc::Sender`) is `Sync`.
    let placed: Vec<(usize, Vec<f32>)> = {
        let slices = split_mut_by_lens(data, &lens);
        pool.install(|| {
            slices
                .into_par_iter()
                .zip(buckets.par_iter())
                .zip(offsets.par_iter())
                .map(|((dst, src), &offset)| {
                    dst.copy_from_slice(src);
                    (offset, dst.to_vec())
                })
                .collect()
        })
    };
    for (offset, values) in placed {
        let indices: Vec<usize> = (offset..offset + values.len()).collect();
        let _ = tx.send(SortEvent::Progress { indices, values });
    }

    let _ = tx.send(SortEvent::Done(SortRunStats {
        elapsed: start.elapsed(),
        elements: n,
        algorithm: "gsa-hybrid-sample-bitonic",
        threads_used,
        gpu_used: true,
        gpu_device: Some(gpu_ctx.device_name.clone()),
        bucket_multiplier: Some(bucket_multiplier),
    }));
}

fn partition_into_buckets(data: &[f32], num_buckets: usize) -> Vec<Vec<f32>> {
    let n = data.len();
    if num_buckets <= 1 {
        return vec![data.to_vec()];
    }

    let sample_count = (num_buckets * 16).min(n);
    let mut rng = rand::thread_rng();
    let mut sample: Vec<f32> = (0..sample_count)
        .map(|_| data[rng.gen_range(0..n)])
        .collect();
    sample.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut pivots = Vec::with_capacity(num_buckets.saturating_sub(1));
    for i in 1..num_buckets {
        let idx = ((i * sample.len()) / num_buckets).min(sample.len().saturating_sub(1));
        pivots.push(sample[idx]);
    }

    let bucket_idx: Vec<usize> = data
        .par_iter()
        .map(|&v| pivots.partition_point(|&p| p <= v))
        .collect();

    let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); num_buckets];
    for (i, &v) in data.iter().enumerate() {
        buckets[bucket_idx[i]].push(v);
    }
    buckets
}

/// Split `data` into consecutive mutable sub-slices of the given lengths.
fn split_mut_by_lens<'a>(data: &'a mut [f32], lens: &[usize]) -> Vec<&'a mut [f32]> {
    let mut rest = data;
    let mut out = Vec::with_capacity(lens.len());
    for &l in lens {
        let (a, b) = rest.split_at_mut(l);
        out.push(a);
        rest = b;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threadpool::build_pool;
    use std::sync::mpsc::channel;

    fn run_sort(mut data: Vec<f32>) -> Vec<f32> {
        let pool_info = build_pool();
        let (tx, rx) = channel();
        gsa_sort(&mut data, &pool_info.pool, None, &tx);
        while let Ok(event) = rx.try_recv() {
            if let SortEvent::Done(_) = event {
                break;
            }
        }
        data
    }

    #[test]
    fn empty_array() {
        assert_eq!(run_sort(vec![]), Vec::<f32>::new());
    }

    #[test]
    fn single_element() {
        assert_eq!(run_sort(vec![42.0]), vec![42.0]);
    }

    #[test]
    fn duplicates() {
        let result = run_sort(vec![3.0, 1.0, 3.0, 1.0, 2.0, 3.0]);
        assert_eq!(result, vec![1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn already_sorted() {
        let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(run_sort(input.clone()), input);
    }

    #[test]
    fn reverse_sorted() {
        let input: Vec<f32> = (0..1000).map(|i| (1000 - i) as f32).collect();
        let mut expected = input.clone();
        expected.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(run_sort(input), expected);
    }

    #[test]
    fn random_large_cpu_fallback() {
        let mut rng = rand::thread_rng();
        let input: Vec<f32> = (0..20_000).map(|_| rng.gen_range(-1e6..1e6)).collect();
        let mut expected = input.clone();
        expected.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(run_sort(input), expected);
    }
}
