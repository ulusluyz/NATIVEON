#[cfg(test)]
mod tests {
    use nativeon::compute::{wave_associative_mix, register_state_update, cpu_reference_wam};
    use nativeon::emulator::WaveMode;

    #[test]
    fn test_wam_matches_cpu_reference() {
        let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let gpu_sim_res = wave_associative_mix(&input, WaveMode::Wave32);
        let cpu_ref_res = cpu_reference_wam(&input, 32);

        for i in 0..32 {
            let diff = (gpu_sim_res[i] - cpu_ref_res[i]).abs();
            assert!(diff < 1e-5, "Mismatch at lane {}: gpu={}, cpu={}", i, gpu_sim_res[i], cpu_ref_res[i]);
        }
    }

    #[test]
    fn test_register_state_update() {
        let mut state = vec![1.0f32; 32];
        let input = vec![0.5f32; 32];
        let decay = 0.9f32;
        let weight = 0.2f32;

        register_state_update(&mut state, &input, decay, weight, WaveMode::Wave32);

        // Expected: 1.0 * 0.9 + 0.5 * 0.2 = 0.9 + 0.1 = 1.0
        for i in 0..32 {
            assert!((state[i] - 1.0).abs() < 1e-5);
        }
    }
}
