//! Parallel LSD radix sort for `f32`.
//!
//! Comparison sorts are fundamentally limited to O(n log n); radix sort
//! sidesteps the comparison lower bound entirely by sorting on fixed-width
//! integer keys, giving O(n) time over 4 passes of 8-bit digits. GSA uses
//! this for its CPU-only path (arrays too small to justify GPU dispatch,
//! or no GPU present) instead of a comparison sort.
//!
//! `f32` isn't directly radix-sortable because IEEE-754 bit patterns don't
//! preserve numeric order across the sign boundary. The standard fix
//! (Herf, "Radix Sort Revisited") is a monotonic bit transform: flip the
//! sign bit for positive numbers, flip every bit for negative numbers.
//! That maps `f32` onto `u32` in a way that unsigned integer comparison
//! matches the original floating-point order, so an ordinary unsigned
//! radix sort on the transformed keys sorts the floats. Assumes finite,
//! non-NaN input (true for GSA's use case: bar heights / plain numbers).
//!
//! The counting/prefix-sum/scatter passes are parallelized across the
//! rayon pool: each thread computes a local histogram over its chunk,
//! chunk-relative offsets are derived from those histograms sequentially
//! (cheap: `num_chunks * 256` integers), and the scatter writes each
//! chunk's keys into disjoint target ranges in parallel.

use rayon::prelude::*;

const RADIX_BITS: u32 = 8;
const RADIX_SIZE: usize = 1 << RADIX_BITS;
const RADIX_MASK: u32 = (RADIX_SIZE as u32) - 1;
const PASSES: u32 = 32 / RADIX_BITS;

#[inline]
fn to_key(f: f32) -> u32 {
    let bits = f.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

#[inline]
fn from_key(bits: u32) -> f32 {
    let bits = if bits & 0x8000_0000 != 0 {
        bits & 0x7FFF_FFFF
    } else {
        !bits
    };
    f32::from_bits(bits)
}

/// A raw pointer wrapper used only to let independent, provably-disjoint
/// writes into the same backing allocation cross a rayon closure boundary.
/// Safety: callers must guarantee every write index is touched by exactly
/// one thread across the whole parallel scatter.
struct DisjointWrite(*mut u32);
unsafe impl Sync for DisjointWrite {}
unsafe impl Send for DisjointWrite {}

impl DisjointWrite {
    // A method call (rather than a field access) forces the closure below
    // to capture all of `self` as one unit instead of Rust 2021's
    // per-field precise capture pulling out the bare `*mut u32`, which
    // would silently drop the `unsafe impl Sync` this type provides.
    #[inline]
    fn ptr(&self) -> *mut u32 {
        self.0
    }
}

/// Sort `data` in place using a 4-pass parallel LSD radix sort. Runs on
/// whichever rayon pool is current when called; wrap in `pool.install`.
pub fn radix_sort_f32_parallel(data: &mut [f32]) {
    let n = data.len();
    if n < 2 {
        return;
    }

    let mut keys: Vec<u32> = data.par_iter().map(|&f| to_key(f)).collect();
    let mut scratch: Vec<u32> = vec![0u32; n];

    let num_threads = rayon::current_num_threads().max(1);
    let chunk_len = n.div_ceil(num_threads).max(1);

    for pass in 0..PASSES {
        let shift = pass * RADIX_BITS;

        // 1. Per-chunk histograms, computed in parallel.
        let chunk_hists: Vec<[usize; RADIX_SIZE]> = keys
            .par_chunks(chunk_len)
            .map(|chunk| {
                let mut hist = [0usize; RADIX_SIZE];
                for &k in chunk {
                    hist[((k >> shift) & RADIX_MASK) as usize] += 1;
                }
                hist
            })
            .collect();
        let num_chunks = chunk_hists.len();

        // 2. Turn per-chunk histograms into per-chunk starting offsets:
        // for each digit, chunk c's slice starts right after the global
        // count of smaller digits plus every earlier chunk's count of
        // this digit. Sequential, but only num_chunks * 256 integers of
        // work, cheap relative to the parallel scatter below.
        let mut chunk_offsets = vec![[0usize; RADIX_SIZE]; num_chunks];
        {
            let mut running = 0usize;
            for digit in 0..RADIX_SIZE {
                for (c, hist) in chunk_hists.iter().enumerate() {
                    chunk_offsets[c][digit] = running;
                    running += hist[digit];
                }
            }
        }

        // 3. Parallel scatter: each chunk writes its keys into disjoint
        // ranges of `scratch`, since chunk_offsets guarantees no two
        // chunks ever target the same slot.
        let dest = DisjointWrite(scratch.as_mut_ptr());
        keys.par_chunks(chunk_len)
            .zip(chunk_offsets.into_par_iter())
            .for_each(|(chunk, mut offsets)| {
                let dest_ptr = dest.ptr();
                for &k in chunk {
                    let digit = ((k >> shift) & RADIX_MASK) as usize;
                    let target = offsets[digit];
                    offsets[digit] += 1;
                    // SAFETY: `target` is unique across all chunks/threads
                    // by construction of `chunk_offsets`.
                    unsafe {
                        *dest_ptr.add(target) = k;
                    }
                }
            });

        std::mem::swap(&mut keys, &mut scratch);
    }

    data.par_iter_mut()
        .zip(keys.par_iter())
        .for_each(|(d, &k)| *d = from_key(k));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(mut input: Vec<f32>) {
        let mut expected = input.clone();
        expected.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        radix_sort_f32_parallel(&mut input);
        assert_eq!(input, expected);
    }

    #[test]
    fn empty() {
        check(vec![]);
    }

    #[test]
    fn single() {
        check(vec![1.0]);
    }

    #[test]
    fn mixed_signs_and_duplicates() {
        check(vec![3.0, -1.0, 0.0, -0.0, 5.5, -5.5, 3.0, -1.0, 100.0, -100.0]);
    }

    #[test]
    fn already_sorted_and_reverse() {
        let asc: Vec<f32> = (0..5000).map(|i| i as f32).collect();
        check(asc.clone());
        let desc: Vec<f32> = asc.into_iter().rev().collect();
        check(desc);
    }

    #[test]
    fn random_large() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let input: Vec<f32> = (0..123_457).map(|_| rng.gen_range(-1e9..1e9)).collect();
        check(input);
    }
}
