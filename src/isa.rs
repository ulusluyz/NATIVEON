//! NATIVEON AMDGPU ISA Definitions, Encoder & Decoder
//!
//! Provides direct encoding/decoding and structured representation for AMDGPU ISA instructions
//! targeting Wave32/Wave64 execution (RDNA3 gfx1100 / CDNA3 gfx940).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Register {
    /// Scalar Register (SGPR)
    SGPR(u16),
    /// Vector Register (VGPR)
    VGPR(u16),
    /// Accumulator Register (AGPR - CDNA)
    AGPR(u16),
    /// Inline Constant / Literal
    Literal(u32),
    /// Special EXEC mask register
    EXEC,
    /// Special VCC register
    VCC,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    // Vector Operations (VOP1 / VOP2 / VOP3)
    VAddF32,
    VMulF32,
    VFmaF32,
    VBfiB32,
    VAndB32,
    VOrB32,
    VXorB32,
    VReadLaneB32,

    // Data Share Operations (DS)
    DsBPermuteB32,
    DsPermuteB32,
    DsReadB32,
    DsWriteB32,

    // Scalar Operations (SOP1 / SOP2)
    SMovB32,
    SAddU32,
    SAndB64,
    SAndSaveExecB64,

    // NOP / Control
    SWaitcnt,
    SNop,
    SEndpgm,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    pub opcode: Opcode,
    pub dst: Option<Register>,
    pub src0: Option<Register>,
    pub src1: Option<Register>,
    pub src2: Option<Register>,
    pub literal: Option<u32>,
}

impl Instruction {
    pub fn new(opcode: Opcode) -> Self {
        Self {
            opcode,
            dst: None,
            src0: None,
            src1: None,
            src2: None,
            literal: None,
        }
    }

    pub fn with_dst(mut self, reg: Register) -> Self {
        self.dst = Some(reg);
        self
    }

    pub fn with_src0(mut self, reg: Register) -> Self {
        self.src0 = Some(reg);
        self
    }

    pub fn with_src1(mut self, reg: Register) -> Self {
        self.src1 = Some(reg);
        self
    }

    pub fn with_src2(mut self, reg: Register) -> Self {
        self.src2 = Some(reg);
        self
    }

    pub fn with_literal(mut self, val: u32) -> Self {
        self.literal = Some(val);
        self
    }

    /// Encodes instruction into raw 32-bit or 64-bit AMDGPU machine code bytes (Simulated binary representation)
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let op_code_val: u32 = match self.opcode {
            Opcode::VAddF32 => 0x01,
            Opcode::VMulF32 => 0x02,
            Opcode::VFmaF32 => 0x03,
            Opcode::VBfiB32 => 0x04,
            Opcode::VAndB32 => 0x05,
            Opcode::VOrB32 => 0x06,
            Opcode::VXorB32 => 0x07,
            Opcode::VReadLaneB32 => 0x08,
            Opcode::DsBPermuteB32 => 0x10,
            Opcode::DsPermuteB32 => 0x11,
            Opcode::DsReadB32 => 0x12,
            Opcode::DsWriteB32 => 0x13,
            Opcode::SMovB32 => 0x20,
            Opcode::SAddU32 => 0x21,
            Opcode::SAndB64 => 0x22,
            Opcode::SAndSaveExecB64 => 0x23,
            Opcode::SWaitcnt => 0x30,
            Opcode::SNop => 0x00,
            Opcode::SEndpgm => 0xFF,
        };
        bytes.extend_from_slice(&op_code_val.to_le_bytes());
        if let Some(lit) = self.literal {
            bytes.extend_from_slice(&lit.to_le_bytes());
        }
        bytes
    }
}
