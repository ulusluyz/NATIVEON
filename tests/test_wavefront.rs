#[cfg(test)]
mod tests {
    use nativeon::isa::{Instruction, Opcode, Register};
    use nativeon::emulator::{ExecutionEngine, Wavefront, WaveMode};

    #[test]
    fn test_wavefront_vadd() {
        let mut wave = Wavefront::new(WaveMode::Wave32);
        let mut engine = ExecutionEngine::new();

        // Initialize VGPR1 = 3.5, VGPR2 = 2.5 across all lanes
        for lane in 0..32 {
            wave.set_vgpr_f32(lane, 1, 3.5);
            wave.set_vgpr_f32(lane, 2, 2.5);
        }

        let program = vec![
            Instruction::new(Opcode::VAddF32)
                .with_dst(Register::VGPR(3))
                .with_src0(Register::VGPR(1))
                .with_src1(Register::VGPR(2)),
            Instruction::new(Opcode::SEndpgm),
        ];

        engine.execute_program(&mut wave, &program);

        for lane in 0..32 {
            assert_eq!(wave.get_vgpr_f32(lane, 3), 6.0);
        }
    }

    #[test]
    fn test_inter_lane_ds_bpermute() {
        let mut wave = Wavefront::new(WaveMode::Wave32);
        let mut engine = ExecutionEngine::new();

        // Setup reverse lane shuffle: lane i wants data from lane (31 - i)
        for lane in 0..32 {
            wave.vgpr[lane][1] = (31 - lane) as u32; // addr_lane
            wave.set_vgpr_f32(lane, 2, lane as f32 * 10.0); // src_data
        }

        let program = vec![
            Instruction::new(Opcode::DsBPermuteB32)
                .with_dst(Register::VGPR(3))
                .with_src0(Register::VGPR(1))
                .with_src1(Register::VGPR(2)),
            Instruction::new(Opcode::SEndpgm),
        ];

        engine.execute_program(&mut wave, &program);

        for lane in 0..32 {
            let expected = (31 - lane) as f32 * 10.0;
            assert_eq!(wave.get_vgpr_f32(lane, 3), expected);
        }
    }
}
