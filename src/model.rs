//! NATIVEON Hardware-Native AI Architecture (WaveState SIMD Network)
//!
//! Implements a novel recurrent non-Transformer architecture built directly on
//! NATIVEON compute primitives:
//! - Wave Associative Mixing (WAM)
//! - Register-Resident State Update (RRSU)

use crate::compute::{register_state_update, wave_associative_mix};
use crate::emulator::WaveMode;

#[derive(Debug, Clone)]
pub struct WaveStateConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize, // Must be multiple of wave_size (32 or 64)
    pub wave_mode: WaveMode,
}

#[derive(Debug, Clone)]
pub struct WaveStateModel {
    pub config: WaveStateConfig,
    pub token_embedding: Vec<Vec<f32>>, // [vocab_size, hidden_dim]
    pub decay_params: Vec<f32>,         // [hidden_dim]
    pub input_weights: Vec<f32>,        // [hidden_dim]
    pub output_proj: Vec<Vec<f32>>,      // [vocab_size, hidden_dim]
}

impl WaveStateModel {
    pub fn new(config: WaveStateConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;

        // Initialize deterministic pseudo-random weights for reproducible learning
        let mut token_embedding = vec![vec![0.0f32; hidden_dim]; vocab_size];
        for v in 0..vocab_size {
            for h in 0..hidden_dim {
                token_embedding[v][h] = (((v * 37 + h * 17) % 100) as f32 - 50.0) / 100.0;
            }
        }

        let decay_params = vec![0.85f32; hidden_dim];
        let input_weights = vec![0.25f32; hidden_dim];

        let mut output_proj = vec![vec![0.0f32; hidden_dim]; vocab_size];
        for v in 0..vocab_size {
            for h in 0..hidden_dim {
                output_proj[v][h] = (((v * 13 + h * 29) % 100) as f32 - 50.0) / 100.0;
            }
        }

        Self {
            config,
            token_embedding,
            decay_params,
            input_weights,
            output_proj,
        }
    }

    /// Single sequence step forward pass using NATIVEON compute primitives
    pub fn step(&self, token_id: usize, state: &mut [f32]) -> Vec<f32> {
        let x = &self.token_embedding[token_id % self.config.vocab_size];

        // 1. Register-Resident State Update (RRSU)
        for chunk in 0..(self.config.hidden_dim / (self.config.wave_mode as usize)) {
            let offset = chunk * (self.config.wave_mode as usize);
            let wave_len = self.config.wave_mode as usize;
            let decay = self.decay_params[offset];
            let weight = self.input_weights[offset];

            register_state_update(
                &mut state[offset..(offset + wave_len)],
                &x[offset..(offset + wave_len)],
                decay,
                weight,
                self.config.wave_mode,
            );
        }

        // 2. Wave Associative Mixing (WAM)
        let mixed_state = wave_associative_mix(state, self.config.wave_mode);

        // 3. Compute output logits over vocabulary
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for v in 0..self.config.vocab_size {
            let mut dot = 0.0f32;
            for h in 0..self.config.hidden_dim {
                dot += mixed_state[h] * self.output_proj[v][h];
            }
            logits[v] = dot;
        }

        logits
    }
}
