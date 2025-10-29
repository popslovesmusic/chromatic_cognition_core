//! Comprehensive Dream Pool Validation Experiment with Statistical Analysis
//!
//! This example runs the full A/B test with detailed statistical analysis
//! and generates comprehensive reports.
//!
//! Run with:
//! ```
//! cargo run --example dream_validation_full --release
//! ```

use chromatic_cognition_core::data::DatasetConfig;
use chromatic_cognition_core::dream::simple_pool::PoolConfig;
use chromatic_cognition_core::dream::experiment::{
    ExperimentConfig, ExperimentHarness, SeedingStrategy,
};
use chromatic_cognition_core::dream::analysis::{compare_experiments, generate_report};
use chromatic_cognition_core::ChromaticNativeSolver;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Dream Pool Validation Experiment - Full Analysis            ║");
    println!("║   Validation Experiment Specification: Retrieval Hypothesis    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Shared configuration
    let dataset_config = DatasetConfig {
        tensor_size: (16, 16, 4),
        noise_level: 0.15,
        samples_per_class: 50,
        seed: 42,
    };

    let pool_config = PoolConfig {
        max_size: 500,
        coherence_threshold: 0.65,
        retrieval_limit: 3,
        use_hnsw: true,
        memory_budget_mb: Some(500),
    };

    let num_epochs = 40;
    let dream_iterations = 8;

    println!("Configuration:");
    println!("  Tensor Size: {:?}", dataset_config.tensor_size);
    println!("  Samples per Class: {}", dataset_config.samples_per_class);
    println!("  Total Samples: {}", dataset_config.samples_per_class * 10);
    println!("  Epochs: {}", num_epochs);
    println!("  Dream Iterations: {}", dream_iterations);
    println!("  Pool Coherence Threshold: {}\n", pool_config.coherence_threshold);

    // ========================================================================
    // Group A: Control (Random Noise)
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Group A: Control (Random Noise Seeding)                    │");
    println!("└─────────────────────────────────────────────────────────────┘");

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

    print!("Running experiment... ");
    std::io::stdout().flush().unwrap();
    let result_a = harness_a.run();
    println!("Done!");

    println!("Results:");
    println!("  Final Accuracy: {:.4}", result_a.final_accuracy);
    println!("  Convergence Epoch: {:?}", result_a.convergence_epoch);
    println!("  Total Time: {}ms", result_a.total_elapsed_ms);
    println!(
        "  Mean Coherence: {:.4}\n",
        result_a
            .epoch_metrics
            .iter()
            .map(|m| m.mean_coherence)
            .sum::<f64>()
            / result_a.epoch_metrics.len() as f64
    );

    // ========================================================================
    // Group B: Test (Retrieval-Based)
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Group B: Test (Retrieval-Based Seeding)                    │");
    println!("└─────────────────────────────────────────────────────────────┘");

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

    print!("Running experiment... ");
    std::io::stdout().flush().unwrap();
    let result_b = harness_b.run();
    println!("Done!");

    println!("Results:");
    println!("  Final Accuracy: {:.4}", result_b.final_accuracy);
    println!("  Convergence Epoch: {:?}", result_b.convergence_epoch);
    println!("  Total Time: {}ms", result_b.total_elapsed_ms);

    if let Some(stats) = harness_b.pool_stats() {
        println!("  Pool Size: {}", stats.count);
        println!("  Pool Mean Coherence: {:.4}", stats.mean_coherence);
        println!("  Pool Mean Energy: {:.4}", stats.mean_energy);
    }

    println!(
        "  Mean Coherence: {:.4}\n",
        result_b
            .epoch_metrics
            .iter()
            .map(|m| m.mean_coherence)
            .sum::<f64>()
            / result_b.epoch_metrics.len() as f64
    );

    // ========================================================================
    // Statistical Comparison
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Statistical Analysis                                        │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let comparison = compare_experiments(&result_a, &result_b, 0.01);
    let report = generate_report(&comparison);
    println!("{}", report);

    // ========================================================================
    // Save Results
    // ========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ Saving Results                                              │");
    println!("└─────────────────────────────────────────────────────────────┘");

    std::fs::create_dir_all("logs").expect("Failed to create logs directory");

    // Save raw results
    let json_a = serde_json::to_string_pretty(&result_a).expect("Failed to serialize Group A");
    let mut file_a =
        File::create("logs/validation_group_a.json").expect("Failed to create file");
    file_a.write_all(json_a.as_bytes()).expect("Failed to write");
    println!("✓ Group A results: logs/validation_group_a.json");

    let json_b = serde_json::to_string_pretty(&result_b).expect("Failed to serialize Group B");
    let mut file_b =
        File::create("logs/validation_group_b.json").expect("Failed to create file");
    file_b.write_all(json_b.as_bytes()).expect("Failed to write");
    println!("✓ Group B results: logs/validation_group_b.json");

    // Save comparison
    let json_comp =
        serde_json::to_string_pretty(&comparison).expect("Failed to serialize comparison");
    let mut file_comp =
        File::create("logs/validation_comparison.json").expect("Failed to create file");
    file_comp
        .write_all(json_comp.as_bytes())
        .expect("Failed to write");
    println!("✓ Comparison: logs/validation_comparison.json");

    // Save human-readable report
    let mut report_file =
        File::create("logs/validation_report.txt").expect("Failed to create report file");
    report_file
        .write_all(report.as_bytes())
        .expect("Failed to write report");
    println!("✓ Report: logs/validation_report.txt");

    // ========================================================================
    // Generate CSV for plotting
    // ========================================================================
    let mut csv = String::from("epoch,group_a_accuracy,group_b_accuracy,group_a_coherence,group_b_coherence\n");
    for (a, b) in result_a.epoch_metrics.iter().zip(result_b.epoch_metrics.iter()) {
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            a.epoch, a.validation_accuracy, b.validation_accuracy, a.mean_coherence, b.mean_coherence
        ));
    }

    let mut csv_file =
        File::create("logs/validation_metrics.csv").expect("Failed to create CSV file");
    csv_file.write_all(csv.as_bytes()).expect("Failed to write CSV");
    println!("✓ Metrics CSV: logs/validation_metrics.csv");

    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ Experiment Complete                                         │");
    println!("└─────────────────────────────────────────────────────────────┘");

    // Final decision
    if comparison.is_significant {
        println!("\n🎉 SUCCESS: Retrieval hypothesis VALIDATED!");
        println!("   ✓ Improvement: {:.2}%", comparison.accuracy_improvement_pct);
        println!("   ✓ Statistically significant (p < {})", comparison.significance_level);
        println!("\n   → RECOMMENDATION: Proceed to Phase 2");
        println!("     • Implement SQLite persistence");
        println!("     • Add FFT spectral analysis");
        println!("     • Develop full Dream Pool specification");
    } else if comparison.accuracy_improvement > 0.0 {
        println!("\n⚠️  MARGINAL: Hypothesis weakly supported");
        println!("   • Improvement: {:.2}%", comparison.accuracy_improvement_pct);
        println!("   • Below significance threshold");
        println!("\n   → RECOMMENDATION: Investigate further");
        println!("     • Tune hyperparameters");
        println!("     • Increase dataset size");
        println!("     • Analyze failure modes");
    } else {
        println!("\n❌ NEGATIVE: Hypothesis not validated");
        println!("   • No improvement observed");
        println!("\n   → RECOMMENDATION: Defer Dream Pool");
        println!("     • Focus on core solver optimization");
        println!("     • Consider alternative approaches");
    }
}
