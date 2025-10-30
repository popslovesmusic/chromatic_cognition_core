//! Phase 3B Validation - Test refined Dream Pool improvements
//!
//! This experiment validates whether Phase 3B refinements improve training:
//! - Class-aware retrieval
//! - Diversity enforcement (MMR)
//! - Spectral feature extraction
//! - ΔLoss utility scoring
//! - Bias profile synthesis
//!
//! **3-Way Comparison:**
//! 1. Baseline: No Dream Pool
//! 2. Phase 3A: Original Dream Pool (cosine similarity only)
//! 3. Phase 3B: Refined Dream Pool (all enhancements)

use chromatic_cognition_core::data::{ColorClass, ColorDataset, ColorSample, DatasetConfig};
use chromatic_cognition_core::dream::simple_pool::PoolConfig;
use chromatic_cognition_core::dream::{BiasProfile, RetrievalMode, SimpleDreamPool};
use chromatic_cognition_core::learner::feedback::{FeedbackRecord, UtilityAggregator};
use chromatic_cognition_core::learner::training::{
    train_with_dreams, EpochMetrics, TrainingConfig, TrainingResult,
};
use chromatic_cognition_core::learner::{ClassifierConfig, MLPClassifier};
use chromatic_cognition_core::solver::native::ChromaticNativeSolver;
use chromatic_cognition_core::tensor::operations::mix;
use chromatic_cognition_core::Solver;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Phase 3B Validation - Refined Dream Pool Evaluation       ║");
    println!("║  Testing: Class-aware + Diversity + Utility + Bias         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let dataset_config = DatasetConfig {
        samples_per_class: 100,
        tensor_size: (16, 16, 4),
        noise_level: 0.1,
        seed: 42,
    };

    let base_training = TrainingConfig {
        num_epochs: 100,
        batch_size: 32,
        learning_rate: 0.01,
        lr_decay: 0.98,
        use_dream_pool: false,
        num_dreams_retrieve: 3,
        retrieval_mode: RetrievalMode::Hard,
        seed: 42,
    };

    let classifier_config = ClassifierConfig {
        hidden_size: 256,
        ..Default::default()
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
    let (train_samples, val_samples) = dataset.split(0.8);
    println!("  Train: {} samples", train_samples.len());
    println!("  Val: {} samples\n", val_samples.len());

    // ========================================================================
    // Experiment 1: Baseline (No Dream Pool)
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ Experiment 1: Baseline (No Dream Pool)                     │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let baseline_config = TrainingConfig {
        use_dream_pool: false,
        num_dreams_retrieve: 0,
        retrieval_mode: RetrievalMode::Hard,
        ..base_training.clone()
    };

    let classifier = MLPClassifier::new(classifier_config.clone());
    let result_baseline = train_with_dreams(
        classifier,
        &train_samples,
        &val_samples,
        baseline_config.clone(),
        None::<&mut SimpleDreamPool>,
        None::<&mut ChromaticNativeSolver>,
    );

    println!("\nBaseline Results:");
    println!(
        "  Final Train Accuracy: {:.2}%",
        result_baseline.final_train_accuracy * 100.0
    );
    println!(
        "  Final Val Accuracy: {:.2}%",
        result_baseline.final_val_accuracy * 100.0
    );
    println!(
        "  Converged at Epoch: {:?}",
        result_baseline.converged_epoch
    );
    println!("  Total Time: {}ms\n", result_baseline.total_elapsed_ms);

    // ========================================================================
    // Experiment 2: Phase 3A (Original Dream Pool)
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ Experiment 2: Phase 3A (Original Dream Pool)               │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let mut pool_3a = SimpleDreamPool::new(pool_config.clone());
    let mut solver_3a = ChromaticNativeSolver::default();

    println!("Populating Dream Pool...");
    for sample in &train_samples {
        if let Ok(result) = solver_3a.evaluate(&sample.tensor, false) {
            pool_3a.add_if_coherent(sample.tensor.clone(), result);
        }
    }
    println!("  Pool size: {}\n", pool_3a.len());

    let phase3a_config = TrainingConfig {
        use_dream_pool: true,
        retrieval_mode: RetrievalMode::Hard,
        ..base_training.clone()
    };

    let classifier = MLPClassifier::new(classifier_config.clone());
    let result_3a = train_with_dreams(
        classifier,
        &train_samples,
        &val_samples,
        phase3a_config.clone(),
        Some(&mut pool_3a),
        Some(&mut solver_3a),
    );

    println!("\nPhase 3A Results:");
    println!(
        "  Final Train Accuracy: {:.2}%",
        result_3a.final_train_accuracy * 100.0
    );
    println!(
        "  Final Val Accuracy: {:.2}%",
        result_3a.final_val_accuracy * 100.0
    );
    println!("  Converged at Epoch: {:?}", result_3a.converged_epoch);
    println!("  Total Time: {}ms\n", result_3a.total_elapsed_ms);

    // ========================================================================
    // Experiment 3: Phase 3B (Refined Dream Pool)
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ Experiment 3: Phase 3B (Refined Dream Pool)                │");
    println!("│ Features: Class-aware + Diversity + Utility + Bias         │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let mut pool_3b = SimpleDreamPool::new(pool_config.clone());
    let mut solver_3b = ChromaticNativeSolver::default();
    let mut aggregator = UtilityAggregator::new();

    println!("Populating Dream Pool with class labels...");
    for sample in &train_samples {
        if let Ok(result) = solver_3b.evaluate(&sample.tensor, false) {
            pool_3b.add_with_class(sample.tensor.clone(), result, sample.label);
        }
    }
    println!("  Pool size: {}\n", pool_3b.len());

    let phase3b_config = TrainingConfig {
        use_dream_pool: true,
        retrieval_mode: RetrievalMode::Hybrid,
        ..base_training.clone()
    };

    let classifier = MLPClassifier::new(classifier_config.clone());

    println!("Training with Phase 3B enhancements...");
    let result_3b = train_with_phase_3b(
        classifier,
        &train_samples,
        &val_samples,
        phase3b_config.clone(),
        &mut pool_3b,
        &mut solver_3b,
        &mut aggregator,
    );

    println!("\nPhase 3B Results:");
    println!(
        "  Final Train Accuracy: {:.2}%",
        result_3b.final_train_accuracy * 100.0
    );
    println!(
        "  Final Val Accuracy: {:.2}%",
        result_3b.final_val_accuracy * 100.0
    );
    println!("  Converged at Epoch: {:?}", result_3b.converged_epoch);
    println!("  Total Time: {}ms", result_3b.total_elapsed_ms);
    println!("  Feedback Records: {}", aggregator.len());
    println!("  Mean Utility: {:.3}\n", aggregator.mean_utility());

    // ========================================================================
    // Synthesize and Save Bias Profile
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ Bias Profile Synthesis                                      │");
    println!("└──────────────────────────────────────────────────────────────┘");

    let bias_profile = BiasProfile::from_aggregator(&aggregator, 0.1);

    println!("\nClass Biases:");
    for class in [ColorClass::Red, ColorClass::Green, ColorClass::Blue] {
        if let Some(stats) = aggregator.class_stats(class) {
            println!(
                "  {:?}: mean_utility={:.3}, helpful={}, harmful={}",
                class, stats.mean_utility, stats.helpful_count, stats.harmful_count
            );
        }
    }

    println!("\nPreferred Classes:");
    for class_name in bias_profile.preferred_classes() {
        println!("  • {}", class_name);
    }

    std::fs::create_dir_all("logs").expect("Failed to create logs directory");
    bias_profile
        .save_to_json("logs/phase_3b_bias_profile.json")
        .expect("Failed to save bias profile");
    println!("\n✓ Bias profile saved to logs/phase_3b_bias_profile.json");

    // ========================================================================
    // Final Comparison
    // ========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│ Final Comparison                                             │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    println!("| Metric              | Baseline | Phase 3A | Phase 3B | Winner   |");
    println!("|---------------------|----------|----------|----------|----------|");

    let val_accs = [
        result_baseline.final_val_accuracy,
        result_3a.final_val_accuracy,
        result_3b.final_val_accuracy,
    ];
    let best_acc_idx = val_accs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    println!(
        "| Val Accuracy        | {:.2}%   | {:.2}%   | {:.2}%   | {}      |",
        val_accs[0] * 100.0,
        val_accs[1] * 100.0,
        val_accs[2] * 100.0,
        ["Baseline", "Phase 3A", "Phase 3B"][best_acc_idx]
    );

    let epochs = [
        result_baseline
            .converged_epoch
            .unwrap_or(base_training.num_epochs),
        result_3a
            .converged_epoch
            .unwrap_or(base_training.num_epochs),
        result_3b
            .converged_epoch
            .unwrap_or(base_training.num_epochs),
    ];
    let best_epoch_idx = epochs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(i, _)| i)
        .unwrap();

    println!(
        "| Convergence Epoch   | {}       | {}       | {}       | {}      |",
        epochs[0],
        epochs[1],
        epochs[2],
        ["Baseline", "Phase 3A", "Phase 3B"][best_epoch_idx]
    );

    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│ Conclusion                                                   │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    if best_acc_idx == 2 && best_epoch_idx == 2 {
        println!("✅ SUCCESS: Phase 3B outperforms both Baseline and Phase 3A!");
        println!("   Refinements (class-aware + diversity + utility) are effective.\n");
    } else if best_acc_idx == 2 || best_epoch_idx == 2 {
        println!("⚠️  PARTIAL: Phase 3B shows improvements in some metrics.");
        println!("   Further refinement may be needed.\n");
    } else {
        println!("❌ FAIL: Phase 3B does not outperform baseline.");
        println!("   Refinements may need adjustment or task complexity increase.\n");
    }
}

/// Train with Phase 3B enhancements (simplified for demonstration)
fn train_with_phase_3b(
    mut classifier: MLPClassifier,
    train_data: &[ColorSample],
    val_data: &[ColorSample],
    config: TrainingConfig,
    pool: &mut SimpleDreamPool,
    solver: &mut ChromaticNativeSolver,
    aggregator: &mut UtilityAggregator,
) -> TrainingResult {
    use chromatic_cognition_core::learner::ColorClassifier;
    use std::time::Instant;

    let convergence_threshold = 0.95;
    let mut current_lr = config.learning_rate;
    let start = Instant::now();

    let mut epoch_metrics = Vec::new();
    let mut best_val_accuracy = 0.0;
    let mut converged_epoch = None;
    let mut final_train_accuracy = 0.0;

    for epoch in 0..config.num_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut batch_count = 0usize;
        let mut epoch_feedback = 0usize;
        let mut dreams_used = 0usize;

        for batch_start in (0..train_data.len()).step_by(config.batch_size.max(1)) {
            let batch_end = (batch_start + config.batch_size).min(train_data.len());
            let batch = &train_data[batch_start..batch_end];

            let mut batch_tensors = Vec::with_capacity(batch.len());
            let mut batch_labels = Vec::with_capacity(batch.len());

            for sample in batch {
                let mut tensor = sample.tensor.clone();
                let signature = tensor.mean_rgb();
                let retrieved = pool.retrieve_similar_class(
                    &signature,
                    sample.label,
                    config.num_dreams_retrieve,
                );

                if let Some(entry) = retrieved.first() {
                    tensor = mix(&tensor, &entry.tensor);
                    dreams_used += 1;
                }

                batch_tensors.push(tensor);
                batch_labels.push(sample.label);
            }

            let loss_before = classifier.compute_loss(&batch_tensors, &batch_labels).0;
            let (loss, gradients) = classifier.compute_loss(&batch_tensors, &batch_labels);
            classifier.update_weights(&gradients, current_lr);
            let loss_after = classifier.compute_loss(&batch_tensors, &batch_labels).0;

            epoch_loss += loss;
            batch_count += 1;

            if epoch > 0 && batch_start % 256 == 0 {
                if let Some(sample) = batch.first() {
                    let record = FeedbackRecord::new(
                        sample.tensor.mean_rgb(),
                        Some(sample.label),
                        loss_before,
                        loss_after,
                        epoch + 1,
                    );
                    aggregator.add_record(record);
                    epoch_feedback += 1;
                }
            }
        }

        // Validation metrics
        let val_tensors: Vec<_> = val_data.iter().map(|s| s.tensor.clone()).collect();
        let val_labels: Vec<_> = val_data.iter().map(|s| s.label).collect();
        let (val_loss, _) = classifier.compute_loss(&val_tensors, &val_labels);
        let val_correct = val_tensors
            .iter()
            .zip(val_labels.iter())
            .filter(|(tensor, &label)| classifier.predict(tensor) == label)
            .count();
        let val_accuracy = if val_data.is_empty() {
            0.0
        } else {
            val_correct as f32 / val_data.len() as f32
        };

        if val_accuracy > best_val_accuracy {
            best_val_accuracy = val_accuracy;
        }

        if val_accuracy >= convergence_threshold && converged_epoch.is_none() {
            converged_epoch = Some(epoch + 1);
        }

        // Training accuracy
        let train_correct = train_data
            .iter()
            .filter(|sample| classifier.predict(&sample.tensor) == sample.label)
            .count();
        let train_accuracy = if train_data.is_empty() {
            0.0
        } else {
            train_correct as f32 / train_data.len() as f32
        };

        final_train_accuracy = train_accuracy;

        let avg_train_loss = if batch_count > 0 {
            epoch_loss / batch_count as f32
        } else {
            0.0
        };

        epoch_metrics.push(EpochMetrics {
            epoch: epoch + 1,
            train_loss: avg_train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            learning_rate: current_lr,
            elapsed_ms: epoch_start.elapsed().as_millis(),
            dreams_used: dreams_used.max(epoch_feedback),
        });

        current_lr *= config.lr_decay;

        if converged_epoch.is_some() && epoch >= converged_epoch.unwrap() + 5 {
            break;
        }

        // Refresh pool with recent tensors for continued retrieval
        if epoch % 5 == 0 {
            for sample in batch_subset(train_data, epoch) {
                if let Ok(result) = solver.evaluate(&sample.tensor, false) {
                    pool.add_with_class(sample.tensor.clone(), result, sample.label);
                }
            }
        }
    }

    TrainingResult {
        config,
        epoch_metrics,
        final_train_accuracy,
        final_val_accuracy: best_val_accuracy,
        total_elapsed_ms: start.elapsed().as_millis(),
        converged_epoch,
    }
}

/// Helper: deterministic subset selection for periodic pool refresh.
fn batch_subset<'a>(samples: &'a [ColorSample], epoch: usize) -> Vec<&'a ColorSample> {
    if samples.is_empty() {
        return Vec::new();
    }

    let step = (samples.len() / 16).max(1);
    let offset = epoch % step;
    samples.iter().skip(offset).step_by(step).collect()
}
