#[cfg(test)]
mod tests {
    use nativeon::emulator::WaveMode;
    use nativeon::model::{WaveStateConfig, WaveStateModel};
    use nativeon::train::{CheckpointManager, Generator, Trainer};

    #[test]
    fn test_training_decreases_loss() {
        let config = WaveStateConfig {
            vocab_size: 16,
            hidden_dim: 32,
            wave_mode: WaveMode::Wave32,
        };
        let mut model = WaveStateModel::new(config);
        let trainer = Trainer::new(0.05);

        let sequence = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];

        let initial_loss = trainer.train_sequence(&mut model, &sequence);

        let mut final_loss = initial_loss;
        for _ in 0..50 {
            final_loss = trainer.train_sequence(&mut model, &sequence);
        }

        assert!(
            final_loss < initial_loss,
            "Loss should decrease during training: initial={}, final={}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_checkpoint_save_and_load() {
        let config = WaveStateConfig {
            vocab_size: 16,
            hidden_dim: 32,
            wave_mode: WaveMode::Wave32,
        };
        let model = WaveStateModel::new(config);

        let file_path = "/tmp/test_nativeon_model.bin";
        CheckpointManager::save(&model, file_path).expect("Failed to save checkpoint");

        let loaded_model = CheckpointManager::load(file_path).expect("Failed to load checkpoint");

        assert_eq!(loaded_model.config.vocab_size, model.config.vocab_size);
        assert_eq!(loaded_model.config.hidden_dim, model.config.hidden_dim);

        for v in 0..model.config.vocab_size {
            for h in 0..model.config.hidden_dim {
                assert_eq!(
                    loaded_model.token_embedding[v][h],
                    model.token_embedding[v][h]
                );
            }
        }
    }

    #[test]
    fn test_autoregressive_inference() {
        let config = WaveStateConfig {
            vocab_size: 16,
            hidden_dim: 32,
            wave_mode: WaveMode::Wave32,
        };
        let model = WaveStateModel::new(config);

        let prompt = vec![1, 2];
        let gen = Generator::generate(&model, &prompt, 5);

        assert_eq!(gen.len(), 7); // 2 prompt + 5 generated
        assert_eq!(gen[0], 1);
        assert_eq!(gen[1], 2);
    }
}
