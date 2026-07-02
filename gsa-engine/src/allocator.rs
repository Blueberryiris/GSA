//! Fixed-size scratch memory arena.
//!
//! GSA reserves a fixed ~4 GB block of RAM on startup regardless of the
//! input array size. This is intentional showcase behavior: the point of
//! GSA is to demonstrate that the machine's full memory bandwidth/capacity
//! is available to it, not to allocate proportionally to the workload.
//! The arena backs GPU staging buffers and thread-local sort workspaces
//! via a simple bump allocator.
//!
//! macOS's memory compressor is aggressive: an idle process's resident set
//! gets compressed (and can drop out of `RSS` as reported by `ps`/`top`)
//! within seconds, even for pages that were genuinely written once. To
//! keep the ~4 GB claim honestly visible in Activity Monitor for the life
//! of the process, a background thread periodically re-touches the arena
//! (see [`ScratchArena::spawn_keepalive`]).

use rayon::prelude::*;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const PAGE_SIZE: usize = 4096;

pub struct ScratchArena {
    buffer: Vec<AtomicU8>,
    offset: AtomicUsize,
}

impl ScratchArena {
    /// Allocate and physically commit `size_bytes` of RAM by filling every
    /// page with pseudo-random bytes, in parallel. Writing forces the OS to
    /// back each page with real physical memory instead of leaving it as a
    /// lazily-mapped zero page; filling with high-entropy (rather than
    /// mostly-zero) content also makes the pages harder for the memory
    /// compressor to shrink.
    pub fn new(size_bytes: usize) -> Self {
        let buffer: Vec<AtomicU8> = (0..size_bytes).map(|_| AtomicU8::new(0)).collect();
        let num_chunks = rayon::current_num_threads().max(1);
        let chunk_len = (buffer.len() / num_chunks).max(PAGE_SIZE);
        buffer
            .par_chunks(chunk_len)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                // A cheap xorshift-style hash gives each byte enough
                // apparent entropy to be hard to compress without needing
                // a real CSPRNG; this is memory-pressure filler, not
                // secret data.
                let mut state = (chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
                for byte in chunk.iter() {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    byte.store(state as u8, Ordering::Relaxed);
                }
            });
        Self {
            buffer,
            offset: AtomicUsize::new(0),
        }
    }

    /// Spawn a background thread that periodically re-touches one byte per
    /// page across the whole arena, forever, keeping the reservation
    /// resident (and out of the memory compressor) for as long as the
    /// process runs.
    pub fn spawn_keepalive(self: &Arc<Self>) {
        let arena = Arc::clone(self);
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(2));
            let mut i = 0;
            while i < arena.buffer.len() {
                let prev = arena.buffer[i].load(Ordering::Relaxed);
                arena.buffer[i].store(prev.wrapping_add(1), Ordering::Relaxed);
                i += PAGE_SIZE;
            }
        });
    }

    /// Reserve a byte range from the arena for scratch use. Returns `None`
    /// if the arena is exhausted (callers should fall back to a heap
    /// allocation in that case; the arena is a showcase optimization, not
    /// a hard requirement for correctness).
    pub fn alloc(&self, size: usize) -> Option<std::ops::Range<usize>> {
        let start = self.offset.fetch_add(size, Ordering::SeqCst);
        if start + size > self.buffer.len() {
            self.offset.fetch_sub(size, Ordering::SeqCst);
            return None;
        }
        Some(start..start + size)
    }

    pub fn reset(&self) {
        self.offset.store(0, Ordering::SeqCst);
    }

    pub fn committed_bytes(&self) -> usize {
        self.buffer.len()
    }
}
