use nativeon::benchmark::BenchmarkRunner;

fn main() {
    println!("============================================================");
    println!("       NATIVEON AMD GPU COMPUTE BENCHMARK SUITE             ");
    println!("============================================================");

    let wam_res = BenchmarkRunner::bench_wam(10_000);
    println!("Primitive: {}", wam_res.primitive_name);
    println!("Iterations: {}", wam_res.iterations);
    println!("Avg Latency: {:.2} ns", wam_res.avg_time_ns);
    println!("Throughput: {:.2} SIMD ops/sec", wam_res.ops_per_sec);
    println!("------------------------------------------------------------");

    let rrsu_res = BenchmarkRunner::bench_rrsu(10_000);
    println!("Primitive: {}", rrsu_res.primitive_name);
    println!("Iterations: {}", rrsu_res.iterations);
    println!("Avg Latency: {:.2} ns", rrsu_res.avg_time_ns);
    println!("Throughput: {:.2} SIMD ops/sec", rrsu_res.ops_per_sec);
    println!("============================================================");
}
