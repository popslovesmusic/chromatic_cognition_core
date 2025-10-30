//! A/B testing harness for Dream Pool validation experiments
//!
//! This module implements the validation experiment specified in
//! "Validation Experiment Specification: Retrieval Hypothesis"

use crate::data::{ColorDataset, ColorSample, DatasetConfig};
use crate::dream::simple_pool::{PoolConfig, PoolStats};
use crate::dream::SimpleDreamPool;
use crate::solver::Solver;
use crate::tensor::{operations::mix, ChromaticTensor};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Seeding strategy for the experiment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedingStrategy {
    /// Group A: Random noise or zero tensor (control)
    RandomNoise,
    /// Group B: Retrieval-based seeding from dream pool (test)
    RetrievalBased,
}

/// Configuration for a single experimental run
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Seeding strategy to use
    pub strategy: SeedingStrategy,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of solver iterations per dream cycle
    pub dream_iterations: usize,
    /// Pool configuration (only used for RetrievalBased strategy)
    pub pool_config: PoolConfig,
    /// Dataset configuration
    pub dataset_config: DatasetConfig,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            strategy: SeedingStrategy::RandomNoise,
            num_epochs: 50,
            batch_size: 10,
            dream_iterations: 10,
            pool_config: PoolConfig::default(),
            dataset_config: DatasetConfig::default(),
            seed: 42,
        }
    }
}

/// Metrics for a single training step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    pub epoch: usize,
    pub step: usize,
    pub energy: f64,
    pub coherence: f64,
    pub violation: f64,
    pub elapsed_ms: u128,
}

/// Metrics for a single epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub mean_energy: f64,
    pub mean_coherence: f64,
    pub mean_violation: f64,
    pub validation_accuracy: f64,
    pub elapsed_ms: u128,
}

/// Complete experiment results
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentResult {
    pub strategy: String,
    pub step_metrics: Vec<StepMetrics>,
    pub epoch_metrics: Vec<EpochMetrics>,
    pub final_accuracy: f64,
    pub convergence_epoch: Option<usize>,
    pub total_elapsed_ms: u128,
}

/// A/B test harness for dream pool validation
pub struct ExperimentHarness<S: Solver> {
    config: ExperimentConfig,
    solver: S,
    pool: Option<SimpleDreamPool>,
}

impl<S: Solver> ExperimentHarness<S> {
    /// Create a new experiment harness
    pub fn new(config: ExperimentConfig, solver: S) -> Self {
        let pool = if config.strategy == SeedingStrategy::RetrievalBased {
            Some(SimpleDreamPool::new(config.pool_config.clone()))
        } else {
            None
        };

        Self {
            config,
            solver,
            pool,
        }
    }

    /// Run the complete experiment
    pub fn run(&mut self) -> ExperimentResult {
        let start_time = Instant::now();

        // Generate dataset
        let dataset = ColorDataset::generate(self.config.dataset_config.clone());
        let (train_samples, val_samples) = dataset.split(0.8);

        let mut step_metrics = Vec::new();
        let mut epoch_metrics = Vec::new();

        for epoch in 0..self.config.num_epochs {
            let epoch_start = Instant::now();
            let epoch_stats = self.run_epoch(epoch, &train_samples, &mut step_metrics);

            // Compute validation accuracy
            let val_accuracy = self.validate(&val_samples);

            epoch_metrics.push(EpochMetrics {
                epoch,
                mean_energy: epoch_stats.0,
                mean_coherence: epoch_stats.1,
                mean_violation: epoch_stats.2,
                validation_accuracy: val_accuracy,
                elapsed_ms: epoch_start.elapsed().as_millis(),
            });
        }

        let final_accuracy = epoch_metrics
            .last()
            .map(|m| m.validation_accuracy)
            .unwrap_or(0.0);
        let convergence_epoch = self.find_convergence_epoch(&epoch_metrics, final_accuracy);

        ExperimentResult {
            strategy: format!("{:?}", self.config.strategy),
            step_metrics,
            epoch_metrics,
            final_accuracy,
            convergence_epoch,
            total_elapsed_ms: start_time.elapsed().as_millis(),
        }
    }

    /// Run a single training epoch
    fn run_epoch(
        &mut self,
        epoch: usize,
        train_samples: &[ColorSample],
        step_metrics: &mut Vec<StepMetrics>,
    ) -> (f64, f64, f64) {
        let mut sum_energy = 0.0;
        let mut sum_coherence = 0.0;
        let mut sum_violation = 0.0;
        let mut step_count = 0;

        for (batch_idx, sample) in train_samples.iter().enumerate() {
            let step_start = Instant::now();

            // Generate seed tensor based on strategy
            let seed_tensor = self.generate_seed_tensor(&sample.tensor);

            // Run dream cycle (multiple solver iterations)
            let mut current_tensor = seed_tensor;
            let mut final_result = None;

            for _ in 0..self.config.dream_iterations {
                let result = self
                    .solver
                    .evaluate(&current_tensor, false)
                    .expect("Solver evaluation failed");

                final_result = Some(result.clone());

                // Simple update: mix with input (simulation of gradient descent)
                current_tensor = mix(&current_tensor, &sample.tensor);
            }

            let result = final_result.expect("No solver result");

            // Store in pool if using retrieval strategy
            if let Some(ref mut pool) = self.pool {
                pool.add_if_coherent(current_tensor.clone(), result.clone());
            }

            // Record metrics
            sum_energy += result.energy;
            sum_coherence += result.coherence;
            sum_violation += result.violation;
            step_count += 1;

            step_metrics.push(StepMetrics {
                epoch,
                step: batch_idx,
                energy: result.energy,
                coherence: result.coherence,
                violation: result.violation,
                elapsed_ms: step_start.elapsed().as_millis(),
            });
        }

        let mean_energy = sum_energy / step_count as f64;
        let mean_coherence = sum_coherence / step_count as f64;
        let mean_violation = sum_violation / step_count as f64;

        (mean_energy, mean_coherence, mean_violation)
    }

    /// Generate seed tensor based on strategy
    fn generate_seed_tensor(&self, input_tensor: &ChromaticTensor) -> ChromaticTensor {
        let (rows, cols, layers) = input_tensor.dims();

        match self.config.strategy {
            SeedingStrategy::RandomNoise => {
                // Group A: Random noise
                ChromaticTensor::from_seed(
                    self.config.seed + rand::random::<u64>(),
                    rows,
                    cols,
                    layers,
                )
            }
            SeedingStrategy::RetrievalBased => {
                // Group B: Retrieve and blend
                let query_signature = input_tensor.mean_rgb();

                if let Some(ref pool) = self.pool {
                    let similar = pool.retrieve_similar(&query_signature, 3);

                    if similar.is_empty() {
                        // Pool empty, fallback to noise
                        ChromaticTensor::from_seed(
                            self.config.seed + rand::random::<u64>(),
                            rows,
                            cols,
                            layers,
                        )
                    } else {
                        // Blend retrieved tensors
                        let mut seed = ChromaticTensor::from_seed(
                            self.config.seed + rand::random::<u64>(),
                            rows,
                            cols,
                            layers,
                        );

                        for entry in similar {
                            seed = mix(&seed, &entry.tensor);
                        }

                        seed
                    }
                } else {
                    panic!("Pool not initialized for retrieval strategy");
                }
            }
        }
    }

    /// Validate on validation set (dummy implementation - returns coherence as proxy for accuracy)
    fn validate(&mut self, val_samples: &[ColorSample]) -> f64 {
        if val_samples.is_empty() {
            return 0.0;
        }

        let mut total_coherence = 0.0;

        for sample in val_samples.iter().take(50) {
            let result = self
                .solver
                .evaluate(&sample.tensor, false)
                .expect("Validation evaluation failed");
            total_coherence += result.coherence;
        }

        let sample_count = val_samples.len().min(50);
        total_coherence / sample_count as f64
    }

    /// Find the epoch where convergence occurred (90% of final accuracy)
    fn find_convergence_epoch(
        &self,
        epoch_metrics: &[EpochMetrics],
        final_accuracy: f64,
    ) -> Option<usize> {
        let target = final_accuracy * 0.9;

        for metrics in epoch_metrics {
            if metrics.validation_accuracy >= target {
                return Some(metrics.epoch);
            }
        }

        None
    }

    /// Get pool statistics (if using retrieval strategy)
    pub fn pool_stats(&self) -> Option<PoolStats> {
        self.pool.as_ref().map(|p| p.stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChromaticNativeSolver;

    #[test]
    fn test_experiment_control_group() {
        let config = ExperimentConfig {
            strategy: SeedingStrategy::RandomNoise,
            num_epochs: 2,
            batch_size: 5,
            dream_iterations: 3,
            dataset_config: DatasetConfig {
                samples_per_class: 5,
                tensor_size: (8, 8, 2),
                ..Default::default()
            },
            ..Default::default()
        };

        let solver = ChromaticNativeSolver::default();
        let mut harness = ExperimentHarness::new(config, solver);
        let result = harness.run();

        assert_eq!(result.epoch_metrics.len(), 2);
        assert!(result.final_accuracy >= 0.0);
        assert_eq!(result.strategy, "RandomNoise");
    }

    #[test]
    fn test_experiment_retrieval_group() {
        let config = ExperimentConfig {
            strategy: SeedingStrategy::RetrievalBased,
            num_epochs: 2,
            batch_size: 5,
            dream_iterations: 3,
            pool_config: PoolConfig {
                coherence_threshold: 0.0, // Accept all for testing
                ..Default::default()
            },
            dataset_config: DatasetConfig {
                samples_per_class: 5,
                tensor_size: (8, 8, 2),
                ..Default::default()
            },
            ..Default::default()
        };

        let solver = ChromaticNativeSolver::default();
        let mut harness = ExperimentHarness::new(config, solver);
        let result = harness.run();

        assert_eq!(result.epoch_metrics.len(), 2);
        assert!(result.final_accuracy >= 0.0);
        assert_eq!(result.strategy, "RetrievalBased");

        // Should have populated pool
        let stats = harness.pool_stats().expect("Pool should exist");
        assert!(stats.count > 0);
    }
}
