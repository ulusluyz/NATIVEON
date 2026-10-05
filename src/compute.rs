//! NATIVEON Compute Primitives
//!
//! Provides the core low-level compute primitives designed specifically
//! for AMD GPU wavefront characteristics:
//! 1. Wave Associative Mixing (WAM) - Intra-wave inter-lane shuffles
//! 2. Register-Resident State Update (RRSU) - In-register recurrent state dynamics
//! 3. Global State Broadcast (GSB) - Scalar register to vector lane broadcasting
//! 4. Quantized Bit Mask Activation (QBMA) - Bitwise fast activation functions

use crate::emulator::{ExecutionEngine, Wavefront, WaveMode};
use crate::isa::{Instruction, Opcode, Register};

/// Wave Associative Mixing (WAM)
/// mixes lane values within a wavefront using associative shift-permute operations.
pub fn wave_associative_mix(input: &[f32], wave_mode: WaveMode) -> Vec<f32> {
    let num_lanes = wave_mode as usize;
    let mut output = vec![0.0f32; input.len()];

    for chunk_idx in 0..(input.len() / num_lanes) {
        let mut wave = Wavefront::new(wave_mode);
        let mut engine = ExecutionEngine::new();

        let offset = chunk_idx * num_lanes;
        for lane in 0..num_lanes {
            let addr = ((lane + 1) % num_lanes) as u32; // Rotate right by 1 lane
            wave.vgpr[lane][1] = addr;
            wave.set_vgpr_f32(lane, 2, input[offset + lane]);
        }

        let program = vec![
            Instruction::new(Opcode::DsBPermuteB32)
                .with_dst(Register::VGPR(3))
                .with_src0(Register::VGPR(1))
                .with_src1(Register::VGPR(2)),
            Instruction::new(Opcode::VFmaF32)
                .with_dst(Register::VGPR(4))
                .with_src0(Register::VGPR(3)) // rotated neighbor
                .with_src1(Register::VGPR(2)) // self
                .with_src2(Register::VGPR(2)),
            Instruction::new(Opcode::SEndpgm),
        ];

        engine.execute_program(&mut wave, &program);

        for lane in 0..num_lanes {
            output[offset + lane] = wave.get_vgpr_f32(lane, 4);
        }
    }

    output
}

/// Register-Resident State Update (RRSU)
/// Updates hidden state vector in-place inside registers using SIMD operations:
/// state_next = decay * state + input_weight * input
pub fn register_state_update(
    state: &mut [f32],
    input: &[f32],
    decay: f32,
    input_weight: f32,
    wave_mode: WaveMode,
) {
    let num_lanes = wave_mode as usize;
    for chunk_idx in 0..(state.len() / num_lanes) {
        let mut wave = Wavefront::new(wave_mode);
        let mut engine = ExecutionEngine::new();
        let offset = chunk_idx * num_lanes;

        wave.set_sgpr_f32(0, decay);
        wave.set_sgpr_f32(1, input_weight);

        for lane in 0..num_lanes {
            wave.set_vgpr_f32(lane, 0, state[offset + lane]); // current state
            wave.set_vgpr_f32(lane, 1, input[offset + lane]); // input vector
        }

        let program = vec![
            Instruction::new(Opcode::VMulF32)
                .with_dst(Register::VGPR(2))
                .with_src0(Register::VGPR(0))
                .with_src1(Register::SGPR(0)), // state * decay
            Instruction::new(Opcode::VFmaF32)
                .with_dst(Register::VGPR(3))
                .with_src0(Register::VGPR(1))
                .with_src1(Register::SGPR(1))
                .with_src2(Register::VGPR(2)), // (input * input_weight) + (state * decay)
            Instruction::new(Opcode::SEndpgm),
        ];

        engine.execute_program(&mut wave, &program);

        for lane in 0..num_lanes {
            state[offset + lane] = wave.get_vgpr_f32(lane, 3);
        }
    }
}

/// CPU Reference Implementation of Wave Associative Mixing for accuracy testing
pub fn cpu_reference_wam(input: &[f32], num_lanes: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for chunk in 0..(input.len() / num_lanes) {
        let offset = chunk * num_lanes;
        for lane in 0..num_lanes {
            let neighbor_val = input[offset + ((lane + 1) % num_lanes)];
            let self_val = input[offset + lane];
            output[offset + lane] = neighbor_val * self_val + self_val;
        }
    }
    output
}
