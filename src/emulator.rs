//! NATIVEON AMD GPU Wavefront Execution Simulator & Reference Runtime Engine
//!
//! Simulates physical AMD GPU Wavefront dynamics (Wave32 / Wave64),
//! Scalar (SGPR) & Vector (VGPR) register banks, Local Data Share (LDS),
//! and execution of AMDGPU ISA instructions including inter-lane `ds_bpermute_b32` shuffles.

use crate::isa::{Instruction, Opcode, Register};

pub const MAX_SGPRS: usize = 104;
pub const MAX_VGPRS: usize = 256;
pub const LDS_SIZE_BYTES: usize = 65536; // 64 KiB LDS per CU

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveMode {
    Wave32 = 32,
    Wave64 = 64,
}

pub struct Wavefront {
    pub mode: WaveMode,
    pub num_lanes: usize,
    pub sgpr: [u32; MAX_SGPRS],
    pub vgpr: Vec<[u32; MAX_VGPRS]>, // VGPR per lane [lane][reg_idx]
    pub exec_mask: u64,
    pub pc: usize,
    pub finished: bool,
}

impl Wavefront {
    pub fn new(mode: WaveMode) -> Self {
        let num_lanes = mode as usize;
        let exec_mask = if num_lanes == 32 { 0xFFFF_FFFF } else { 0xFFFF_FFFF_FFFF_FFFF };
        Self {
            mode,
            num_lanes,
            sgpr: [0; MAX_SGPRS],
            vgpr: vec![[0; MAX_VGPRS]; num_lanes],
            exec_mask,
            pc: 0,
            finished: false,
        }
    }

    pub fn set_vgpr_f32(&mut self, lane: usize, reg_idx: usize, val: f32) {
        self.vgpr[lane][reg_idx] = val.to_bits();
    }

    pub fn get_vgpr_f32(&self, lane: usize, reg_idx: usize) -> f32 {
        f32::from_bits(self.vgpr[lane][reg_idx])
    }

    pub fn set_sgpr_f32(&mut self, reg_idx: usize, val: f32) {
        self.sgpr[reg_idx] = val.to_bits();
    }

    pub fn get_sgpr_f32(&self, reg_idx: usize) -> f32 {
        f32::from_bits(self.sgpr[reg_idx])
    }
}

pub struct ExecutionEngine {
    pub lds: Vec<u8>,
}

impl ExecutionEngine {
    pub fn new() -> Self {
        Self {
            lds: vec![0; LDS_SIZE_BYTES],
        }
    }

    /// Step a wavefront through a single ISA instruction
    pub fn step(&mut self, wave: &mut Wavefront, instr: &Instruction) {
        if wave.finished {
            return;
        }

        match instr.opcode {
            Opcode::SNop => {},
            Opcode::SEndpgm => {
                wave.finished = true;
            },
            Opcode::SMovB32 => {
                if let (Some(Register::SGPR(dst)), Some(src)) = (instr.dst, instr.src0) {
                    let val = match src {
                        Register::SGPR(s) => wave.sgpr[s as usize],
                        Register::Literal(l) => l,
                        _ => 0,
                    };
                    wave.sgpr[dst as usize] = val;
                }
            },
            Opcode::VAddF32 => {
                if let (Some(Register::VGPR(dst)), Some(src0), Some(src1)) = (instr.dst, instr.src0, instr.src1) {
                    for lane in 0..wave.num_lanes {
                        if (wave.exec_mask & (1 << lane)) != 0 {
                            let v0 = match src0 {
                                Register::VGPR(r) => wave.get_vgpr_f32(lane, r as usize),
                                Register::SGPR(r) => wave.get_sgpr_f32(r as usize),
                                _ => 0.0,
                            };
                            let v1 = match src1 {
                                Register::VGPR(r) => wave.get_vgpr_f32(lane, r as usize),
                                Register::SGPR(r) => wave.get_sgpr_f32(r as usize),
                                _ => 0.0,
                            };
                            wave.set_vgpr_f32(lane, dst as usize, v0 + v1);
                        }
                    }
                }
            },
            Opcode::VMulF32 => {
                if let (Some(Register::VGPR(dst)), Some(src0), Some(src1)) = (instr.dst, instr.src0, instr.src1) {
                    for lane in 0..wave.num_lanes {
                        if (wave.exec_mask & (1 << lane)) != 0 {
                            let v0 = match src0 {
                                Register::VGPR(r) => wave.get_vgpr_f32(lane, r as usize),
                                Register::SGPR(r) => wave.get_sgpr_f32(r as usize),
                                _ => 0.0,
                            };
                            let v1 = match src1 {
                                Register::VGPR(r) => wave.get_vgpr_f32(lane, r as usize),
                                Register::SGPR(r) => wave.get_sgpr_f32(r as usize),
                                _ => 0.0,
                            };
                            wave.set_vgpr_f32(lane, dst as usize, v0 * v1);
                        }
                    }
                }
            },
            Opcode::VFmaF32 => {
                if let (Some(Register::VGPR(dst)), Some(src0), Some(src1), Some(src2)) = (instr.dst, instr.src0, instr.src1, instr.src2) {
                    for lane in 0..wave.num_lanes {
                        if (wave.exec_mask & (1 << lane)) != 0 {
                            let v0 = match src0 {
                                Register::VGPR(r) => wave.get_vgpr_f32(lane, r as usize),
                                Register::SGPR(r) => wave.get_sgpr_f32(r as usize),
                                _ => 0.0,
                            };
                            let v1 = match src1 {
                                Register::VGPR(r) => wave.get_vgpr_f32(lane, r as usize),
                                Register::SGPR(r) => wave.get_sgpr_f32(r as usize),
                                _ => 0.0,
                            };
                            let v2 = match src2 {
                                Register::VGPR(r) => wave.get_vgpr_f32(lane, r as usize),
                                Register::SGPR(r) => wave.get_sgpr_f32(r as usize),
                                _ => 0.0,
                            };
                            wave.set_vgpr_f32(lane, dst as usize, v0 * v1 + v2);
                        }
                    }
                }
            },
            Opcode::DsBPermuteB32 => {
                // Inter-lane Wave Shuffle (ds_bpermute_b32 dst, addr_lane, src_data)
                // Swaps 32-bit values between lanes based on lane indices in addr_lane
                if let (Some(Register::VGPR(dst)), Some(Register::VGPR(addr_reg)), Some(Register::VGPR(src_reg))) = (instr.dst, instr.src0, instr.src1) {
                    let mut temp = vec![0u32; wave.num_lanes];
                    for lane in 0..wave.num_lanes {
                        let target_lane = (wave.vgpr[lane][addr_reg as usize] as usize) % wave.num_lanes;
                        temp[lane] = wave.vgpr[target_lane][src_reg as usize];
                    }
                    for lane in 0..wave.num_lanes {
                        if (wave.exec_mask & (1 << lane)) != 0 {
                            wave.vgpr[lane][dst as usize] = temp[lane];
                        }
                    }
                }
            },
            _ => {},
        }
        wave.pc += 1;
    }

    /// Execute a full sequence of instructions
    pub fn execute_program(&mut self, wave: &mut Wavefront, program: &[Instruction]) {
        while wave.pc < program.len() && !wave.finished {
            let instr = program[wave.pc].clone();
            self.step(wave, &instr);
        }
    }
}
