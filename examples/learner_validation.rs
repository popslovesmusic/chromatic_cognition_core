//! Learner Validation Experiment
//!
//! This example validates the Minimal Viable Learner (MVP) and demonstrates
//! that real gradient descent training works with the color classification task.
//!
//! Validates:
//! 1. Training can actually learn (reaches >90% accuracy)
//! 2. Retrieval-based seeding from Dream Pool helps convergence
//! 3. Proper feedback loop infrastructure for full LEARNER MANIFEST v1.0
//!
//! Run with:
//! ```
//! cargo run --example learner_validation --release
//! ```

use chromatic_cognition_core::data::{ColorDataset, DatasetConfig};
use chromatic_cognition_core::dream::simple_pool::PoolConfig;
use chromatic_cognition_core::dream::{RetrievalMode, SimpleDreamPool};
use chromatic_cognition_core::{
    ChromaticNativeSolver, ClassifierConfig, MLPClassifier, TrainingConfig,
    train_baseline, train_with_dreams,
};
use std::fs::File;
use std::io::Write;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Learner Validation Experiment - Minimal Viable Learner     ║");
    println!("║  Goal: Prove training works & Dream Pool helps convergence  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let dataset_config = DatasetConfig {
        tensor_size: (16, 16, 4),
        noise_level: 0.1,
        samples_per_class: 100, // 1000 total samples
        seed: 42,
    };

    let classifier_config = ClassifierConfig {
        input_size: 16 * 16 * 4 * 3,
        hidden_size: 256,
        output_size: 10,
        seed: 42,
    };

    let pool_config = PoolConfig {
        max_size: 500,
        coherence_threshold: 0.7,
        retrieval_limit: 3,
        use_hnsw: true,
        memory_budget_mb: Some(500),
    };

    println!("Configuration:");
    println!("  Dataset: 1000 samples (100 per class)");
    println!("  Tensor Size: 16×16×4");
    println!("  Model: MLP with 256 hidden units");
    println!("  Pool: 500 max dreams, coherence ≥ 0.7\n");

    // Generate dataset
    println!("Generating dataset...");
    let dataset = ColorDataset::generate(dataset_config);
    let (train_data, val_data) = dataset.split(0.8);
    println!("  Train: {} samples", train_data.len());
    println!("  Val: {} samples\n", val_data.len());

    // ========================================================================
    // Experiment 1: Baseline Training (No Dream Pool)
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ Experiment 1: Baseline Training (No Dream Pool)             │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let baseline_config = TrainingConfig {
        num_epochs: 100,
        batch_size: 32,
        learning_rate: 0.01,
        lr_decay: 0.98,
        use_dream_pool: false,
        num_dreams_retrieve: 0,
        retrieval_mode: RetrievalMode::Hard,
        seed: 42,
    };

    let classifier_baseline = MLPClassifier::new(classifier_config.clone());

    print!("Training baseline model... ");
    std::io::stdout().flush().unwrap();

    let result_baseline = train_baseline(
        classifier_baseline,
        &train_data,
        &val_data,
        baseline_config,
    );

    println!("Done!");
    println!("\nBaseline Results:");
    println!("  Final Train Accuracy: {:.2}%", result_baseline.final_train_accuracy * 100.0);
    println!("  Final Val Accuracy: {:.2}%", result_baseline.final_val_accuracy * 100.0);
    println!("  Converged at Epoch: {:?}", result_baseline.converged_epoch);
    println!("  Total Time: {}ms\n", result_baseline.total_elapsed_ms);

    // ========================================================================
    // Experiment 2: Training with Dream Pool Retrieval
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ Experiment 2: Training with Dream Pool Retrieval            │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let dream_config = TrainingConfig {
        num_epochs: 100,
        batch_size: 32,
        learning_rate: 0.01,
        lr_decay: 0.98,
        use_dream_pool: true,
        num_dreams_retrieve: 3,
        retrieval_mode: RetrievalMode::Hard,
        seed: 42,
    };

    let classifier_dream = MLPClassifier::new(classifier_config.clone());
    let mut pool = SimpleDreamPool::new(pool_config);
    let mut solver = ChromaticNativeSolver::default();

    print!("Training with Dream Pool... ");
    std::io::stdout().flush().unwrap();

    let result_dream = train_with_dreams(
        classifier_dream,
        &train_data,
        &val_data,
        dream_config,
        Some(&mut pool),
        Some(&mut solver),
    );

    println!("Done!");
    println!("\nDream Pool Results:");
    println!("  Final Train Accuracy: {:.2}%", result_dream.final_train_accuracy * 100.0);
    println!("  Final Val Accuracy: {:.2}%", result_dream.final_val_accuracy * 100.0);
    println!("  Converged at Epoch: {:?}", result_dream.converged_epoch);
    println!("  Total Time: {}ms", result_dream.total_elapsed_ms);

    let pool_stats = pool.stats();
    println!("  Pool Size: {}", pool_stats.count);
    println!("  Pool Mean Coherence: {:.4}\n", pool_stats.mean_coherence);

    // ========================================================================
    // Comparison
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ Comparison                                                   │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let accuracy_improvement = result_dream.final_val_accuracy - result_baseline.final_val_accuracy;
    let improvement_pct = (accuracy_improvement / result_baseline.final_val_accuracy) * 100.0;

    println!("Final Validation Accuracy:");
    println!("  Baseline: {:.2}%", result_baseline.final_val_accuracy * 100.0);
    println!("  Dream Pool: {:.2}%", result_dream.final_val_accuracy * 100.0);
    println!("  Improvement: {:.4} ({:.2}%)\n", accuracy_improvement, improvement_pct);

    let convergence_comparison = match (result_baseline.converged_epoch, result_dream.converged_epoch) {
        (Some(baseline_epoch), Some(dream_epoch)) => {
            println!("Convergence (95% accuracy):");
            println!("  Baseline: epoch {}", baseline_epoch);
            println!("  Dream Pool: epoch {}", dream_epoch);
            if dream_epoch < baseline_epoch {
                println!("  Dream Pool converged {} epochs FASTER ✓\n", baseline_epoch - dream_epoch);
            } else if dream_epoch > baseline_epoch {
                println!("  Dream Pool converged {} epochs SLOWER ⚠️\n", dream_epoch - baseline_epoch);
            } else {
                println!("  Same convergence speed\n");
            }
            Some((baseline_epoch, dream_epoch))
        }
        (Some(baseline_epoch), None) => {
            println!("Convergence:");
            println!("  Baseline: epoch {} ✓", baseline_epoch);
            println!("  Dream Pool: Did not converge ⚠️\n");
            None
        }
        (None, Some(dream_epoch)) => {
            println!("Convergence:");
            println!("  Baseline: Did not converge");
            println!("  Dream Pool: epoch {} ✓\n", dream_epoch);
            None
        }
        (None, None) => {
            println!("Convergence:");
            println!("  Neither model converged to 95% accuracy\n");
            None
        }
    };

    // ========================================================================
    // Save Results
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ Saving Results                                               │");
    println!("└──────────────────────────────────────────────────────────────┘");

    std::fs::create_dir_all("logs").expect("Failed to create logs directory");

    // Save JSON results
    let json_baseline = serde_json::to_string_pretty(&result_baseline)
        .expect("Failed to serialize baseline");
    let mut file_baseline = File::create("logs/learner_baseline.json")
        .expect("Failed to create file");
    file_baseline.write_all(json_baseline.as_bytes()).expect("Failed to write");
    println!("✓ Baseline results: logs/learner_baseline.json");

    let json_dream = serde_json::to_string_pretty(&result_dream)
        .expect("Failed to serialize dream pool");
    let mut file_dream = File::create("logs/learner_dream_pool.json")
        .expect("Failed to create file");
    file_dream.write_all(json_dream.as_bytes()).expect("Failed to write");
    println!("✓ Dream Pool results: logs/learner_dream_pool.json");

    // Generate CSV for plotting
    let mut csv = String::from("epoch,baseline_train_acc,baseline_val_acc,dream_train_acc,dream_val_acc,baseline_loss,dream_loss\n");
    for (baseline, dream) in result_baseline.epoch_metrics.iter().zip(result_dream.epoch_metrics.iter()) {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            baseline.epoch,
            baseline.train_accuracy,
            baseline.val_accuracy,
            dream.train_accuracy,
            dream.val_accuracy,
            baseline.train_loss,
            dream.train_loss,
        ));
    }

    let mut csv_file = File::create("logs/learner_comparison.csv")
        .expect("Failed to create CSV file");
    csv_file.write_all(csv.as_bytes()).expect("Failed to write CSV");
    println!("✓ Metrics CSV: logs/learner_comparison.csv");

    // ========================================================================
    // Validation Assessment
    // ========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│ Validation Assessment                                        │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let training_works = result_baseline.final_val_accuracy >= 0.90;
    let retrieval_helps = result_dream.final_val_accuracy > result_baseline.final_val_accuracy;
    let converged_faster = convergence_comparison
        .map(|(baseline, dream)| dream < baseline)
        .unwrap_or(false);

    println!("\n✓ VALIDATION CHECKLIST:\n");

    print!("  [");
    if training_works {
        print!("✓");
    } else {
        print!("✗");
    }
    println!("] Training achieves >90% accuracy");
    println!("      Baseline: {:.2}%", result_baseline.final_val_accuracy * 100.0);

    print!("  [");
    if retrieval_helps {
        print!("✓");
    } else {
        print!("✗");
    }
    println!("] Dream Pool improves final accuracy");
    println!("      Improvement: {:.2}%", improvement_pct);

    print!("  [");
    if converged_faster {
        print!("✓");
    } else {
        print!("✗");
    }
    println!("] Dream Pool accelerates convergence");
    if let Some((baseline, dream)) = convergence_comparison {
        println!("      Baseline: {} epochs, Dream Pool: {} epochs", baseline, dream);
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    if training_works && (retrieval_helps || converged_faster) {
        println!("║  ✓ SUCCESS: Learner validated & Dream Pool hypothesis holds ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!("\n🎉 The Minimal Viable Learner is working!");
        println!("\nNext Steps:");
        println!("  1. ✓ Training algorithm works (gradient descent + cross-entropy)");
        println!("  2. ✓ Dream Pool retrieval integration functional");
        println!("  3. → Ready for full LEARNER MANIFEST v1.0 implementation:");
        println!("       - FFT feature extraction");
        println!("       - Feedback collection (Δloss tracking)");
        println!("       - Bias profile synthesis");
        println!("       - Advanced retrieval modes (euclidean, mixed)");
    } else if training_works && !retrieval_helps {
        println!("║  ⚠️  PARTIAL: Training works, but Dream Pool shows no benefit║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!("\nTraining algorithm is functional, but retrieval hypothesis not validated.");
        println!("\nPossible causes:");
        println!("  - Coherence threshold too high (pool too sparse)");
        println!("  - Dataset too simple (training converges quickly anyway)");
        println!("  - Retrieval strategy needs tuning");
        println!("\nRecommendation: Investigate Dream Pool parameters or try harder task.");
    } else {
        println!("║  ✗ FAILURE: Training did not reach target accuracy          ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!("\nTraining algorithm may need tuning:");
        println!("  - Increase hidden layer size");
        println!("  - Adjust learning rate");
        println!("  - More training epochs");
        println!("  - Better weight initialization");
    }
}
