//! Training loop with Dream Pool integration
//!
//! Implements gradient descent training with optional retrieval-based seeding
//! from the Dream Pool to accelerate convergence.

use crate::data::{ColorDataset, ColorSample};
use crate::dream::SimpleDreamPool;
use crate::learner::classifier::{ColorClassifier, MLPClassifier, ClassifierConfig};
use crate::solver::{Solver, SolverResult};
use crate::tensor::ChromaticTensor;
use crate::tensor::operations::mix;
use serde::{Serialize, Deserialize};
use std::time::Instant;

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub num_epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Learning rate decay per epoch
    pub lr_decay: f32,
    /// Whether to use Dream Pool retrieval
    pub use_dream_pool: bool,
    /// Number of dreams to retrieve per sample
    pub num_dreams_retrieve: usize,
    /// Seed for reproducibility
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 50,
            batch_size: 32,
            learning_rate: 0.01,
            lr_decay: 0.95,
            use_dream_pool: false,
            num_dreams_retrieve: 3,
            seed: 42,
        }
    }
}

/// Training metrics for a single epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    pub train_accuracy: f32,
    pub val_loss: f32,
    pub val_accuracy: f32,
    pub learning_rate: f32,
    pub elapsed_ms: u128,
    pub dreams_used: usize,
}

/// Complete training result
#[derive(Debug, Clone, Serialize)]
pub struct TrainingResult {
    pub config: TrainingConfig,
    pub epoch_metrics: Vec<EpochMetrics>,
    pub final_train_accuracy: f32,
    pub final_val_accuracy: f32,
    pub total_elapsed_ms: u128,
    pub converged_epoch: Option<usize>,
}

/// Compute accuracy on a dataset
fn compute_accuracy<C: ColorClassifier>(
    classifier: &C,
    samples: &[ColorSample],
) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let correct = samples
        .iter()
        .filter(|sample| {
            let predicted = classifier.predict(&sample.tensor);
            predicted == sample.label
        })
        .count();

    correct as f32 / samples.len() as f32
}

/// Augment a tensor with retrieved dreams from the pool
fn augment_with_dreams(
    tensor: &ChromaticTensor,
    pool: &SimpleDreamPool,
    num_dreams: usize,
) -> ChromaticTensor {
    if pool.is_empty() {
        return tensor.clone();
    }

    let query_signature = tensor.mean_rgb();
    let retrieved = pool.retrieve_similar(&query_signature, num_dreams);

    if retrieved.is_empty() {
        return tensor.clone();
    }

    // Mix original tensor with retrieved dreams
    let mut result = tensor.clone();
    for entry in retrieved {
        result = mix(&result, &entry.tensor);
    }

    result
}

/// Train a classifier with optional Dream Pool augmentation
pub fn train_with_dreams<S: Solver>(
    mut classifier: MLPClassifier,
    train_data: &[ColorSample],
    val_data: &[ColorSample],
    config: TrainingConfig,
    mut pool: Option<&mut SimpleDreamPool>,
    mut solver: Option<&mut S>,
) -> TrainingResult {
    let start_time = Instant::now();
    let mut epoch_metrics = Vec::new();

    let mut current_lr = config.learning_rate;
    let mut converged_epoch = None;

    for epoch in 0..config.num_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        let mut dreams_used = 0;

        // Shuffle training data (simple deterministic shuffle based on epoch)
        let mut indices: Vec<usize> = (0..train_data.len()).collect();
        indices.sort_by_key(|&i| (i + epoch * 997) % train_data.len());

        // Process batches
        for batch_start in (0..train_data.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(train_data.len());
            let batch_indices = &indices[batch_start..batch_end];

            let mut batch_tensors = Vec::new();
            let mut batch_labels = Vec::new();

            for &idx in batch_indices {
                let sample = &train_data[idx];

                // Augment with dreams if enabled
                let tensor = if config.use_dream_pool {
                    if let Some(pool) = pool.as_ref() {
                        dreams_used += 1;
                        augment_with_dreams(&sample.tensor, pool, config.num_dreams_retrieve)
                    } else {
                        sample.tensor.clone()
                    }
                } else {
                    sample.tensor.clone()
                };

                batch_tensors.push(tensor);
                batch_labels.push(sample.label);
            }

            // Compute loss and gradients
            let (loss, gradients) = classifier.compute_loss(&batch_tensors, &batch_labels);
            epoch_loss += loss;
            num_batches += 1;

            // Update weights
            classifier.update_weights(&gradients, current_lr);

            // Store augmented tensors in pool if using solver evaluation
            if config.use_dream_pool {
                if let (Some(pool), Some(solver)) = (pool.as_mut(), solver.as_mut()) {
                    for tensor in batch_tensors {
                        if let Ok(result) = solver.evaluate(&tensor, false) {
                            pool.add_if_coherent(tensor, result);
                        }
                    }
                }
            }
        }

        // Compute epoch metrics
        let avg_train_loss = if num_batches > 0 {
            epoch_loss / num_batches as f32
        } else {
            0.0
        };

        let train_accuracy = compute_accuracy(&classifier, train_data);
        let val_accuracy = compute_accuracy(&classifier, val_data);

        // Compute validation loss
        let val_tensors: Vec<_> = val_data.iter().map(|s| s.tensor.clone()).collect();
        let val_labels: Vec<_> = val_data.iter().map(|s| s.label).collect();
        let (val_loss, _) = classifier.compute_loss(&val_tensors, &val_labels);

        epoch_metrics.push(EpochMetrics {
            epoch,
            train_loss: avg_train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            learning_rate: current_lr,
            elapsed_ms: epoch_start.elapsed().as_millis(),
            dreams_used,
        });

        // Check for convergence (95% val accuracy)
        if converged_epoch.is_none() && val_accuracy >= 0.95 {
            converged_epoch = Some(epoch);
        }

        // Learning rate decay
        current_lr *= config.lr_decay;
    }

    let final_train_accuracy = epoch_metrics.last().map(|m| m.train_accuracy).unwrap_or(0.0);
    let final_val_accuracy = epoch_metrics.last().map(|m| m.val_accuracy).unwrap_or(0.0);

    TrainingResult {
        config,
        epoch_metrics,
        final_train_accuracy,
        final_val_accuracy,
        total_elapsed_ms: start_time.elapsed().as_millis(),
        converged_epoch,
    }
}

/// Simple training without Dream Pool (for baseline comparison)
pub fn train_baseline(
    classifier: MLPClassifier,
    train_data: &[ColorSample],
    val_data: &[ColorSample],
    config: TrainingConfig,
) -> TrainingResult {
    let mut config = config;
    config.use_dream_pool = false;

    train_with_dreams::<crate::ChromaticNativeSolver>(
        classifier,
        train_data,
        val_data,
        config,
        None,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColorDataset, DatasetConfig};
    use crate::dream::simple_pool::PoolConfig;
    use crate::ChromaticNativeSolver;

    #[test]
    fn test_compute_accuracy() {
        let config = ClassifierConfig::default();
        let classifier = MLPClassifier::new(config);

        let dataset_config = DatasetConfig {
            tensor_size: (16, 16, 4),
            samples_per_class: 10,
            ..Default::default()
        };
        let dataset = ColorDataset::generate(dataset_config);

        let accuracy = compute_accuracy(&classifier, &dataset.samples);

        // Random classifier should get ~10% accuracy on 10 classes
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }

    #[test]
    fn test_train_baseline() {
        let config = ClassifierConfig {
            hidden_size: 32, // Smaller for faster test
            ..Default::default()
        };
        let classifier = MLPClassifier::new(config);

        let dataset_config = DatasetConfig {
            tensor_size: (16, 16, 4),
            samples_per_class: 5,
            noise_level: 0.05,
            ..Default::default()
        };
        let dataset = ColorDataset::generate(dataset_config);
        let (train_data, val_data) = dataset.split(0.8);

        let train_config = TrainingConfig {
            num_epochs: 3,
            batch_size: 10,
            learning_rate: 0.1,
            lr_decay: 1.0,
            use_dream_pool: false,
            ..Default::default()
        };

        let result = train_baseline(classifier, &train_data, &val_data, train_config);

        assert_eq!(result.epoch_metrics.len(), 3);
        assert!(result.final_train_accuracy >= 0.0);
        assert!(result.final_val_accuracy >= 0.0);

        // Training should improve over random (>10%)
        let initial_acc = result.epoch_metrics[0].train_accuracy;
        let final_acc = result.final_train_accuracy;
        assert!(final_acc >= initial_acc);
    }

    #[test]
    fn test_train_with_dream_pool() {
        let config = ClassifierConfig {
            hidden_size: 32,
            ..Default::default()
        };
        let classifier = MLPClassifier::new(config);

        let dataset_config = DatasetConfig {
            tensor_size: (16, 16, 4),
            samples_per_class: 5,
            noise_level: 0.05,
            ..Default::default()
        };
        let dataset = ColorDataset::generate(dataset_config);
        let (train_data, val_data) = dataset.split(0.8);

        let pool_config = PoolConfig {
            coherence_threshold: 0.5,
            max_size: 100,
            ..Default::default()
        };
        let mut pool = SimpleDreamPool::new(pool_config);
        let mut solver = ChromaticNativeSolver::default();

        let train_config = TrainingConfig {
            num_epochs: 3,
            batch_size: 10,
            learning_rate: 0.1,
            lr_decay: 1.0,
            use_dream_pool: true,
            num_dreams_retrieve: 2,
            ..Default::default()
        };

        let result = train_with_dreams(
            classifier,
            &train_data,
            &val_data,
            train_config,
            Some(&mut pool),
            Some(&mut solver),
        );

        assert_eq!(result.epoch_metrics.len(), 3);
        assert!(result.final_val_accuracy >= 0.0);

        // Pool should have accumulated some dreams
        assert!(pool.len() > 0);
    }

    #[test]
    fn test_augment_with_dreams() {
        let pool_config = PoolConfig {
            coherence_threshold: 0.0,
            max_size: 10,
            ..Default::default()
        };
        let mut pool = SimpleDreamPool::new(pool_config);

        // Add some dreams to pool
        let mut solver = ChromaticNativeSolver::default();
        for i in 0..5 {
            let tensor = ChromaticTensor::from_seed(i, 16, 16, 4);
            let result = solver.evaluate(&tensor, false).unwrap();
            pool.add(tensor, result);
        }

        // Augment a tensor
        let query_tensor = ChromaticTensor::from_seed(100, 16, 16, 4);
        let augmented = augment_with_dreams(&query_tensor, &pool, 3);

        // Augmented tensor should have different mean RGB
        assert_ne!(query_tensor.mean_rgb(), augmented.mean_rgb());
    }
}
