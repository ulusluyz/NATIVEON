//! NATIVEON Training Engine & Autoregressive Inference Pipeline
//!
//! Provides sequence learning, cross-entropy loss computation, parameter updates,
//! binary checkpoint serialization, and autoregressive text generation.

use crate::model::WaveStateModel;
use std::fs::File;
use std::io::{Read, Write};

pub struct Trainer {
    pub lr: f32,
}

impl Trainer {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    /// Train model on sequence of token IDs, returning cross-entropy loss
    pub fn train_sequence(&self, model: &mut WaveStateModel, tokens: &[usize]) -> f32 {
        if tokens.len() < 2 {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut state = vec![0.0f32; model.config.hidden_dim];

        for i in 0..(tokens.len() - 1) {
            let input_token = tokens[i];
            let target_token = tokens[i + 1];

            let logits = model.step(input_token, &mut state);

            // Compute Softmax probabilities
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

            // Cross-entropy loss: -log(p_target)
            let target_prob = probs[target_token % model.config.vocab_size].max(1e-7);
            let loss = -target_prob.ln();
            total_loss += loss;

            // Parameter update (Gradient step on output projection & embedding)
            for v in 0..model.config.vocab_size {
                let grad = if v == target_token % model.config.vocab_size {
                    probs[v] - 1.0
                } else {
                    probs[v]
                };

                for h in 0..model.config.hidden_dim {
                    model.output_proj[v][h] -= self.lr * grad * state[h];
                    model.token_embedding[input_token % model.config.vocab_size][h] -=
                        self.lr * grad * model.output_proj[v][h] * 0.1;
                }
            }
        }

        total_loss / ((tokens.len() - 1) as f32)
    }
}

pub struct Generator;

impl Generator {
    /// Generate autoregressive token sequence starting from prompt tokens
    pub fn generate(model: &WaveStateModel, prompt: &[usize], gen_len: usize) -> Vec<usize> {
        let mut generated = prompt.to_vec();
        let mut state = vec![0.0f32; model.config.hidden_dim];

        // Process prompt tokens to build initial state
        let mut current_token = prompt[0];
        for &tok in prompt {
            current_token = tok;
            let _ = model.step(tok, &mut state);
        }

        // Generate new tokens
        for _ in 0..gen_len {
            let logits = model.step(current_token, &mut state);
            // Greedy selection (Argmax)
            let mut best_token = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (v, &val) in logits.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    best_token = v;
                }
            }
            generated.push(best_token);
            current_token = best_token;
        }

        generated
    }
}

pub struct CheckpointManager;

impl CheckpointManager {
    pub const MAGIC_HEADER: &'static [u8; 8] = b"NATIVEON";

    /// Save model checkpoint to binary file format
    pub fn save(model: &WaveStateModel, filepath: &str) -> std::io::Result<()> {
        let mut file = File::create(filepath)?;
        file.write_all(Self::MAGIC_HEADER)?;

        file.write_all(&(model.config.vocab_size as u32).to_le_bytes())?;
        file.write_all(&(model.config.hidden_dim as u32).to_le_bytes())?;
        file.write_all(&(model.config.wave_mode as u32).to_le_bytes())?;

        for v in 0..model.config.vocab_size {
            for h in 0..model.config.hidden_dim {
                file.write_all(&model.token_embedding[v][h].to_le_bytes())?;
            }
        }

        for h in 0..model.config.hidden_dim {
            file.write_all(&model.decay_params[h].to_le_bytes())?;
            file.write_all(&model.input_weights[h].to_le_bytes())?;
        }

        for v in 0..model.config.vocab_size {
            for h in 0..model.config.hidden_dim {
                file.write_all(&model.output_proj[v][h].to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Load model checkpoint from binary file
    pub fn load(filepath: &str) -> std::io::Result<WaveStateModel> {
        let mut file = File::open(filepath)?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)?;

        if &header != Self::MAGIC_HEADER {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid NATIVEON checkpoint magic header",
            ));
        }

        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let vocab_size = u32::from_le_bytes(buf4) as usize;

        file.read_exact(&mut buf4)?;
        let hidden_dim = u32::from_le_bytes(buf4) as usize;

        file.read_exact(&mut buf4)?;
        let wave_mode_val = u32::from_le_bytes(buf4);
        let wave_mode = if wave_mode_val == 64 {
            crate::emulator::WaveMode::Wave64
        } else {
            crate::emulator::WaveMode::Wave32
        };

        let config = crate::model::WaveStateConfig {
            vocab_size,
            hidden_dim,
            wave_mode,
        };

        let mut model = WaveStateModel::new(config);

        for v in 0..vocab_size {
            for h in 0..hidden_dim {
                file.read_exact(&mut buf4)?;
                model.token_embedding[v][h] = f32::from_le_bytes(buf4);
            }
        }

        for h in 0..hidden_dim {
            file.read_exact(&mut buf4)?;
            model.decay_params[h] = f32::from_le_bytes(buf4);
            file.read_exact(&mut buf4)?;
            model.input_weights[h] = f32::from_le_bytes(buf4);
        }

        for v in 0..vocab_size {
            for h in 0..hidden_dim {
                file.read_exact(&mut buf4)?;
                model.output_proj[v][h] = f32::from_le_bytes(buf4);
            }
        }

        Ok(model)
    }
}
