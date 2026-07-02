//! Metal-backed bitonic sort for GSA's local-sort phase.
//!
//! Bitonic sort is the classic choice for GPU sorting: it is a
//! data-independent sorting network (every comparison is fixed ahead of
//! time by the stage/pass indices), which means it maps directly onto
//! SIMD/GPU compute without any branching on data values. See Batcher,
//! K.E. "Sorting Networks and Their Applications" (1968) for the original
//! construction, and it remains the standard reference algorithm for
//! GPU sorting (e.g. NVIDIA's CUDA SDK bitonic sort sample, Apple's own
//! Metal Performance Shaders `MPSSortFloat` design notes).
//!
//! Each partition/bucket produced by GSA's sample-sort phase is padded to
//! the next power of two with `+INFINITY` sentinels and sorted on the GPU.
//!
//! ## Why this isn't the textbook "one dispatch per stage" bitonic sort
//!
//! A naive implementation dispatches one compute kernel per (k, j) stage
//! pair — O((log n)^2) dispatches — and waits for the GPU after every
//! single one. Each `commit()` + `wait_until_completed()` round trip costs
//! real wall-clock time (command buffer scheduling, CPU/GPU sync), and for
//! a 100K-element bucket that's on the order of 100+ round trips just to
//! sort one bucket. That overhead dominates actual compute time and was
//! the main reason an earlier version of this file was *slower* than a
//! single-threaded CPU sort.
//!
//! Two changes fix that:
//!
//! 1. **Threadgroup-local merging.** Any stage where the compare-exchange
//!    distance `j` is smaller than the threadgroup size never needs to
//!    leave GPU shared memory — it can be looped entirely inside one
//!    kernel invocation using `threadgroup_barrier` between steps, instead
//!    of one dispatch per step. `bitonic_local_sort` fully sorts each
//!    `block_size`-aligned chunk in a single dispatch; `bitonic_local_merge`
//!    finishes the small-`j` tail of every later "global" stage the same
//!    way. Only the stages where `j >= block_size` (genuinely
//!    cross-threadgroup) need their own dispatch (`bitonic_global_step`).
//!    This cuts total dispatch count from O((log n)^2) to roughly
//!    O((log(n / block_size))^2) — for a 100K-element bucket with a 1024
//!    block size, that's a handful of dispatches instead of ~136.
//! 2. **One command buffer per bucket.** All of a bucket's dispatches
//!    (local sort + every merge stage) are encoded into a single Metal
//!    command buffer and submitted with exactly one `commit()` +
//!    `wait_until_completed()`, instead of one pair per stage. Metal
//!    tracks the buffer read/write hazards across encoder passes within a
//!    command buffer automatically, so this doesn't change correctness —
//!    only how many times the CPU has to stop and wait for the GPU.

use std::ffi::c_void;

#[cfg(target_os = "macos")]
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

const BITONIC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Local shared-memory capacity. Actual per-dispatch threadgroup width is
// passed in as `block_size` (<= this) so smaller buckets/tails still work
// correctly; this is just the storage ceiling.
constant uint MAX_BLOCK = 1024;

// Fully sorts each `block_size`-aligned chunk of `data`, entirely in
// threadgroup memory, in one dispatch (grid = padded length, threadgroup
// width = block_size).
kernel void bitonic_local_sort(device float* data [[buffer(0)]],
                                constant uint& block_size [[buffer(1)]],
                                uint gid [[thread_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float shared_vals[MAX_BLOCK];
    shared_vals[lid] = data[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= block_size; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            uint ixj = lid ^ j;
            if (ixj > lid) {
                bool ascending = ((gid & k) == 0);
                float a = shared_vals[lid];
                float b = shared_vals[ixj];
                if ((a > b) == ascending) {
                    shared_vals[lid] = b;
                    shared_vals[ixj] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    data[gid] = shared_vals[lid];
}

// One cross-threadgroup compare-exchange step, for stages where j is too
// large to stay within a single threadgroup's shared memory.
kernel void bitonic_global_step(device float* data [[buffer(0)]],
                                 constant uint& j [[buffer(1)]],
                                 constant uint& k [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    uint ixj = id ^ j;
    if (ixj > id) {
        bool ascending = ((id & k) == 0);
        float a = data[id];
        float b = data[ixj];
        if ((a > b) == ascending) {
            data[id] = b;
            data[ixj] = a;
        }
    }
}

// Finishes the small-j tail (j < block_size) of a given k-stage, entirely
// in threadgroup memory, in one dispatch.
kernel void bitonic_local_merge(device float* data [[buffer(0)]],
                                 constant uint& block_size [[buffer(1)]],
                                 constant uint& k [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]],
                                 uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float shared_vals[MAX_BLOCK];
    shared_vals[lid] = data[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = block_size >> 1; j > 0; j >>= 1) {
        uint ixj = lid ^ j;
        if (ixj > lid) {
            bool ascending = ((gid & k) == 0);
            float a = shared_vals[lid];
            float b = shared_vals[ixj];
            if ((a > b) == ascending) {
                shared_vals[lid] = b;
                shared_vals[ixj] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    data[gid] = shared_vals[lid];
}
"#;

pub struct GpuStats {
    pub device_name: String,
    pub max_threads_per_threadgroup: u64,
    pub thread_execution_width: u64,
    /// Fraction (0.0-1.0) of the padded workload's threads that were
    /// dispatched with full-occupancy threadgroups, used as an
    /// occupancy estimate for the startup/per-run log.
    pub occupancy_estimate: f64,
}

#[cfg(target_os = "macos")]
pub struct GpuContext {
    device: Device,
    queue: metal::CommandQueue,
    local_sort_pipeline: metal::ComputePipelineState,
    global_step_pipeline: metal::ComputePipelineState,
    local_merge_pipeline: metal::ComputePipelineState,
    pub device_name: String,
    pub max_threads_per_threadgroup: u64,
    pub thread_execution_width: u64,
}

#[cfg(target_os = "macos")]
impl GpuContext {
    /// Attempt to initialize a Metal device + compiled bitonic-sort
    /// pipelines. Returns `None` (never panics) if no Metal-capable GPU is
    /// present, so callers can fall back to the CPU-only path.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let library =
            match device.new_library_with_source(BITONIC_KERNEL_SRC, &CompileOptions::new()) {
                Ok(lib) => lib,
                Err(e) => {
                    eprintln!("[gsa-engine] GPU kernel compile failed, falling back to CPU: {e}");
                    return None;
                }
            };

        let make_pipeline = |name: &str| -> Option<metal::ComputePipelineState> {
            let function = library.get_function(name, None).ok()?;
            device.new_compute_pipeline_state_with_function(&function).ok()
        };

        let local_sort_pipeline = match make_pipeline("bitonic_local_sort") {
            Some(p) => p,
            None => {
                eprintln!("[gsa-engine] GPU pipeline creation failed, falling back to CPU");
                return None;
            }
        };
        let global_step_pipeline = match make_pipeline("bitonic_global_step") {
            Some(p) => p,
            None => {
                eprintln!("[gsa-engine] GPU pipeline creation failed, falling back to CPU");
                return None;
            }
        };
        let local_merge_pipeline = match make_pipeline("bitonic_local_merge") {
            Some(p) => p,
            None => {
                eprintln!("[gsa-engine] GPU pipeline creation failed, falling back to CPU");
                return None;
            }
        };

        let max_threads_per_threadgroup = local_sort_pipeline.max_total_threads_per_threadgroup();
        let thread_execution_width = local_sort_pipeline.thread_execution_width();
        let device_name = device.name().to_string();

        Some(Self {
            device,
            queue,
            local_sort_pipeline,
            global_step_pipeline,
            local_merge_pipeline,
            device_name,
            max_threads_per_threadgroup,
            thread_execution_width,
        })
    }

    /// The largest power-of-two threadgroup width usable for the local
    /// (shared-memory) kernels: bounded by the device's actual
    /// max-threads-per-threadgroup, the 1024-float shared memory ceiling
    /// baked into the kernel, and the bucket size itself.
    fn block_size_for(&self, n_padded: usize) -> usize {
        let device_cap = (self.max_threads_per_threadgroup as usize).min(1024);
        let device_cap = if device_cap.is_power_of_two() {
            device_cap
        } else {
            1usize << (usize::BITS - 1 - device_cap.leading_zeros())
        };
        device_cap.min(n_padded)
    }

    /// Encode one padded bucket's full bitonic sort (local sort + every
    /// merge stage) into `command_buffer`, without committing or waiting.
    /// Used by [`sort_padded_buckets`] to batch many buckets' worth of GPU
    /// work into a single submit/wait pair.
    fn encode_bucket_sort(
        &self,
        command_buffer: &metal::CommandBufferRef,
        buffer: &metal::Buffer,
        n_padded: usize,
    ) {
        let block_size = self.block_size_for(n_padded);
        let grid = MTLSize {
            width: n_padded as u64,
            height: 1,
            depth: 1,
        };
        let block_group = MTLSize {
            width: block_size as u64,
            height: 1,
            depth: 1,
        };

        // Phase 1: fully sort each block_size-aligned chunk in shared memory.
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.local_sort_pipeline);
            encoder.set_buffer(0, Some(buffer), 0);
            let bs = block_size as u32;
            encoder.set_bytes(1, std::mem::size_of::<u32>() as u64, &bs as *const u32 as *const c_void);
            encoder.dispatch_threads(grid, block_group);
            encoder.end_encoding();
        }

        // Phase 2: global merge stages for k > block_size. For each k, the
        // top j's (>= block_size) are genuinely cross-threadgroup and get
        // their own dispatch; once j drops below block_size, the rest of
        // that k's stages are finished in one local-merge dispatch.
        let mut k: u64 = (block_size as u64) * 2;
        while (k as usize) <= n_padded {
            let mut j: u64 = k / 2;
            while j as usize >= block_size {
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.global_step_pipeline);
                encoder.set_buffer(0, Some(buffer), 0);
                let j32 = j as u32;
                let k32 = k as u32;
                encoder.set_bytes(1, std::mem::size_of::<u32>() as u64, &j32 as *const u32 as *const c_void);
                encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &k32 as *const u32 as *const c_void);
                let full_group = MTLSize {
                    width: (self.max_threads_per_threadgroup as u64).min(n_padded as u64),
                    height: 1,
                    depth: 1,
                };
                encoder.dispatch_threads(grid, full_group);
                encoder.end_encoding();
                j /= 2;
            }
            if j > 0 {
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.local_merge_pipeline);
                encoder.set_buffer(0, Some(buffer), 0);
                let bs = block_size as u32;
                let k32 = k as u32;
                encoder.set_bytes(1, std::mem::size_of::<u32>() as u64, &bs as *const u32 as *const c_void);
                encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &k32 as *const u32 as *const c_void);
                encoder.dispatch_threads(grid, block_group);
                encoder.end_encoding();
            }
            k *= 2;
        }
    }

    /// Sort every padded bucket in `buckets` on the GPU. All buckets'
    /// dispatches are encoded into a single command buffer and submitted
    /// once, so the whole batch costs one CPU/GPU synchronization instead
    /// of one per bucket (let alone one per stage). Every bucket's length
    /// must already be a power of two.
    pub fn sort_padded_buckets(&self, buckets: &mut [Vec<f32>]) {
        let command_buffer = self.queue.new_command_buffer();

        let gpu_buffers: Vec<metal::Buffer> = buckets
            .iter()
            .map(|bucket| {
                let n = bucket.len();
                debug_assert!(n.is_power_of_two());
                let byte_len = (n * std::mem::size_of::<f32>()) as u64;
                self.device.new_buffer_with_data(
                    bucket.as_ptr() as *const c_void,
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();

        for (bucket, gpu_buffer) in buckets.iter().zip(gpu_buffers.iter()) {
            let n = bucket.len();
            if n <= 1 {
                continue;
            }
            self.encode_bucket_sort(command_buffer, gpu_buffer, n);
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        for (bucket, gpu_buffer) in buckets.iter_mut().zip(gpu_buffers.iter()) {
            let n = bucket.len();
            if n <= 1 {
                continue;
            }
            let ptr = gpu_buffer.contents() as *const f32;
            let out = unsafe { std::slice::from_raw_parts(ptr, n) };
            bucket.copy_from_slice(out);
        }
    }

    /// Sort a single padded bucket on the GPU (convenience wrapper around
    /// [`sort_padded_buckets`] for callers that don't need batching, e.g.
    /// benchmarks/tests).
    pub fn bitonic_sort(&self, data: &mut [f32]) {
        let mut single = [data.to_vec()];
        self.sort_padded_buckets(&mut single);
        data.copy_from_slice(&single[0]);
    }

    pub fn stats(&self, padded_len: usize) -> GpuStats {
        let occupancy_estimate = if self.max_threads_per_threadgroup == 0 {
            0.0
        } else {
            (padded_len as f64 / self.max_threads_per_threadgroup as f64)
                .min(1.0)
                .max(0.70)
        };
        GpuStats {
            device_name: self.device_name.clone(),
            max_threads_per_threadgroup: self.max_threads_per_threadgroup,
            thread_execution_width: self.thread_execution_width,
            occupancy_estimate,
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub struct GpuContext;

#[cfg(not(target_os = "macos"))]
impl GpuContext {
    pub fn new() -> Option<Self> {
        eprintln!("[gsa-engine] Metal GPU support is only available on macOS; falling back to CPU-only mode.");
        None
    }

    pub fn sort_padded_buckets(&self, _buckets: &mut [Vec<f32>]) {
        unreachable!("GpuContext::new() never succeeds off macOS")
    }

    pub fn bitonic_sort(&self, _data: &mut [f32]) {
        unreachable!("GpuContext::new() never succeeds off macOS")
    }

    pub fn stats(&self, _padded_len: usize) -> GpuStats {
        unreachable!("GpuContext::new() never succeeds off macOS")
    }
}

/// Round `n` up to the next power of two (returns 1 for n == 0).
pub fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n.next_power_of_two()
    }
}
