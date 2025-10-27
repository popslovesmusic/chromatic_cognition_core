//! Dream Pool Validation Experiment
//!
//! This example implements the A/B testing methodology described in
//! "Validation Experiment Specification: Retrieval Hypothesis"
//!
//! It compares two groups:
//! - Group A (Control): Random noise seeding
//! - Group B (Test): Retrieval-based seeding from SimpleDreamPool
//!
//! Run with:
//! ```
//! cargo run --example dream_validation --release
//! ```

use chromatic_cognition_core::data::DatasetConfig;
use chromatic_cognition_core::dream::simple_pool::PoolConfig;
use chromatic_cognition_core::dream::{ExperimentConfig, ExperimentHarness, SeedingStrategy};
use chromatic_cognition_core::{ChromaticNativeSolver, Solver};
use std::fs::File;
use std::io::Write;

fn main() {
    println!("=== Dream Pool Validation Experiment ===\n");

    // Shared configuration
    let dataset_config = DatasetConfig {
        tensor_size: (16, 16, 4), // Small tensors for fast iteration
        noise_level: 0.1,
        samples_per_class: 50, // 500 total samples
        seed: 42,
    };

    let pool_config = PoolConfig {
        max_size: 200,
        coherence_threshold: 0.7, // Only keep high-coherence dreams
        retrieval_limit: 3,
    };

    let num_epochs = 30;
    let dream_iterations = 5;

    // Group A: Control (Random Noise)
    println!("Running Group A (Control - Random Noise)...");
    let config_a = ExperimentConfig {
        strategy: SeedingStrategy::RandomNoise,
        num_epochs,
        batch_size: 10,
        dream_iterations,
        pool_config: pool_config.clone(),
        dataset_config: dataset_config.clone(),
        seed: 42,
    };

    let solver_a = ChromaticNativeSolver::default();
    let mut harness_a = ExperimentHarness::new(config_a, solver_a);
    let result_a = harness_a.run();

    println!("  Final Accuracy: {:.4}", result_a.final_accuracy);
    println!("  Convergence Epoch: {:?}", result_a.convergence_epoch);
    println!("  Total Time: {}ms\n", result_a.total_elapsed_ms);

    // Group B: Test (Retrieval-Based)
    println!("Running Group B (Test - Retrieval-Based)...");
    let config_b = ExperimentConfig {
        strategy: SeedingStrategy::RetrievalBased,
        num_epochs,
        batch_size: 10,
        dream_iterations,
        pool_config,
        dataset_config,
        seed: 42,
    };

    let solver_b = ChromaticNativeSolver::default();
    let mut harness_b = ExperimentHarness::new(config_b, solver_b);
    let result_b = harness_b.run();

    println!("  Final Accuracy: {:.4}", result_b.final_accuracy);
    println!("  Convergence Epoch: {:?}", result_b.convergence_epoch);
    println!("  Total Time: {}ms", result_b.total_elapsed_ms);

    if let Some(stats) = harness_b.pool_stats() {
        println!("  Pool Size: {}", stats.count);
        println!("  Pool Mean Coherence: {:.4}\n", stats.mean_coherence);
    }

    // Compare results
    println!("=== Comparison ===");
    println!("Group A Final Accuracy: {:.4}", result_a.final_accuracy);
    println!("Group B Final Accuracy: {:.4}", result_b.final_accuracy);

    let accuracy_improvement = result_b.final_accuracy - result_a.final_accuracy;
    let improvement_pct = (accuracy_improvement / result_a.final_accuracy) * 100.0;

    println!("Accuracy Improvement: {:.4} ({:.2}%)", accuracy_improvement, improvement_pct);

    if let (Some(conv_a), Some(conv_b)) = (result_a.convergence_epoch, result_b.convergence_epoch) {
        let epoch_reduction = conv_a as i32 - conv_b as i32;
        println!("Group A Convergence: epoch {}", conv_a);
        println!("Group B Convergence: epoch {}", conv_b);
        println!("Epochs Saved: {}\n", epoch_reduction);
    } else {
        println!("Convergence: A={:?}, B={:?}\n", result_a.convergence_epoch, result_b.convergence_epoch);
    }

    // Compare mean coherence across epochs
    let mean_coherence_a: f64 = result_a.epoch_metrics.iter().map(|m| m.mean_coherence).sum::<f64>()
        / result_a.epoch_metrics.len() as f64;
    let mean_coherence_b: f64 = result_b.epoch_metrics.iter().map(|m| m.mean_coherence).sum::<f64>()
        / result_b.epoch_metrics.len() as f64;

    println!("Mean Coherence (all epochs):");
    println!("  Group A: {:.4}", mean_coherence_a);
    println!("  Group B: {:.4}", mean_coherence_b);
    println!("  Difference: {:.4}\n", mean_coherence_b - mean_coherence_a);

    // Save detailed results to JSON
    println!("Saving results to logs/...");
    std::fs::create_dir_all("logs").expect("Failed to create logs directory");

    let json_a = serde_json::to_string_pretty(&result_a).expect("Failed to serialize Group A");
    let mut file_a = File::create("logs/experiment_group_a.json").expect("Failed to create file");
    file_a.write_all(json_a.as_bytes()).expect("Failed to write");

    let json_b = serde_json::to_string_pretty(&result_b).expect("Failed to serialize Group B");
    let mut file_b = File::create("logs/experiment_group_b.json").expect("Failed to create file");
    file_b.write_all(json_b.as_bytes()).expect("Failed to write");

    println!("Results saved!");
    println!("\n=== Experiment Complete ===");

    // Decision gate
    println!("\n=== Decision Gate ===");
    if improvement_pct > 5.0 && accuracy_improvement > 0.01 {
        println!("✅ HYPOTHESIS VALIDATED");
        println!("   Retrieval-based seeding shows meaningful improvement.");
        println!("   Recommendation: Proceed to Phase 2 (Persistence, FFT)");
    } else if improvement_pct > 0.0 {
        println!("⚠️  HYPOTHESIS WEAKLY SUPPORTED");
        println!("   Small improvement observed, but below significance threshold.");
        println!("   Recommendation: Tune parameters or investigate further");
    } else {
        println!("❌ HYPOTHESIS NOT VALIDATED");
        println!("   No improvement from retrieval-based seeding.");
        println!("   Recommendation: Defer Dream Pool implementation");
    }
}
