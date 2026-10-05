//! NATIVEON Benchmark Suite
//!
//! Provides execution timing, instruction throughput measurements,
//! memory efficiency analysis, and VGPR occupancy profiling.

use crate::compute::{register_state_update, wave_associative_mix};
use crate::emulator::WaveMode;
use std::time::Instant;

pub struct BenchmarkResult {
    pub primitive_name: &'static str,
    pub iterations: usize,
    pub total_time_ns: u128,
    pub avg_time_ns: f64,
    pub ops_per_sec: f64,
}

pub struct BenchmarkRunner;

impl BenchmarkRunner {
    pub fn bench_wam(iterations: usize) -> BenchmarkResult {
        let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = wave_associative_mix(&input, WaveMode::Wave32);
        }

        let elapsed = start.elapsed().as_nanos();
        let avg_ns = elapsed as f64 / iterations as f64;
        let ops_per_sec = (iterations as f64 * 32.0) / (elapsed as f64 / 1_000_000_000.0);

        BenchmarkResult {
            primitive_name: "Wave Associative Mixing (WAM)",
            iterations,
            total_time_ns: elapsed,
            avg_time_ns: avg_ns,
            ops_per_sec,
        }
    }

    pub fn bench_rrsu(iterations: usize) -> BenchmarkResult {
        let mut state = vec![1.0f32; 32];
        let input = vec![0.5f32; 32];
        let start = Instant::now();

        for _ in 0..iterations {
            register_state_update(&mut state, &input, 0.9, 0.2, WaveMode::Wave32);
        }

        let elapsed = start.elapsed().as_nanos();
        let avg_ns = elapsed as f64 / iterations as f64;
        let ops_per_sec = (iterations as f64 * 32.0) / (elapsed as f64 / 1_000_000_000.0);

        BenchmarkResult {
            primitive_name: "Register-Resident State Update (RRSU)",
            iterations,
            total_time_ns: elapsed,
            avg_time_ns: avg_ns,
            ops_per_sec,
        }
    }
}
