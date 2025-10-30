//! Statistical analysis utilities for experiment results
//!
//! Provides tools for analyzing and comparing A/B test results,
//! including statistical significance testing.

use crate::dream::experiment::{EpochMetrics, ExperimentResult};
use serde::{Deserialize, Serialize};

/// Comparison of two experiment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentComparison {
    pub control_strategy: String,
    pub test_strategy: String,
    pub accuracy_improvement: f64,
    pub accuracy_improvement_pct: f64,
    pub convergence_epoch_delta: Option<i32>,
    pub mean_coherence_improvement: f64,
    pub mean_energy_improvement: f64,
    pub is_significant: bool,
    pub significance_level: f64,
}

/// Compute basic statistics for a sequence of values
#[derive(Debug, Clone, Serialize)]
pub struct Statistics {
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

impl Statistics {
    /// Compute statistics from a slice of values
    pub fn from_slice(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                mean: 0.0,
                variance: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                count: 0,
            };
        }

        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;

        let variance = values
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;

        let std_dev = variance.sqrt();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self {
            mean,
            variance,
            std_dev,
            min,
            max,
            count,
        }
    }
}

/// Compare two experiment results
pub fn compare_experiments(
    control: &ExperimentResult,
    test: &ExperimentResult,
    significance_threshold: f64,
) -> ExperimentComparison {
    // Accuracy comparison
    let accuracy_improvement = test.final_accuracy - control.final_accuracy;
    let accuracy_improvement_pct = if control.final_accuracy > 0.0 {
        (accuracy_improvement / control.final_accuracy) * 100.0
    } else {
        0.0
    };

    // Convergence epoch comparison
    let convergence_epoch_delta = match (control.convergence_epoch, test.convergence_epoch) {
        (Some(a), Some(b)) => Some(a as i32 - b as i32),
        _ => None,
    };

    // Mean coherence comparison
    let control_coherence: Vec<f64> = control
        .epoch_metrics
        .iter()
        .map(|m| m.mean_coherence)
        .collect();
    let test_coherence: Vec<f64> = test
        .epoch_metrics
        .iter()
        .map(|m| m.mean_coherence)
        .collect();

    let mean_coherence_control = Statistics::from_slice(&control_coherence).mean;
    let mean_coherence_test = Statistics::from_slice(&test_coherence).mean;
    let mean_coherence_improvement = mean_coherence_test - mean_coherence_control;

    // Mean energy comparison (lower is better)
    let control_energy: Vec<f64> = control
        .epoch_metrics
        .iter()
        .map(|m| m.mean_energy)
        .collect();
    let test_energy: Vec<f64> = test.epoch_metrics.iter().map(|m| m.mean_energy).collect();

    let mean_energy_control = Statistics::from_slice(&control_energy).mean;
    let mean_energy_test = Statistics::from_slice(&test_energy).mean;
    let mean_energy_improvement = mean_energy_control - mean_energy_test; // Note: reversed (lower is better)

    // Simple significance test: improvement must be > threshold and positive
    let is_significant =
        accuracy_improvement > significance_threshold && accuracy_improvement > 0.0;

    ExperimentComparison {
        control_strategy: control.strategy.clone(),
        test_strategy: test.strategy.clone(),
        accuracy_improvement,
        accuracy_improvement_pct,
        convergence_epoch_delta,
        mean_coherence_improvement,
        mean_energy_improvement,
        is_significant,
        significance_level: significance_threshold,
    }
}

/// Compute t-test statistic for two independent samples
///
/// Returns (t_statistic, degrees_of_freedom)
/// Note: This is Welch's t-test (unequal variances)
pub fn welch_t_test(sample1: &[f64], sample2: &[f64]) -> (f64, f64) {
    let stats1 = Statistics::from_slice(sample1);
    let stats2 = Statistics::from_slice(sample2);

    if stats1.count == 0 || stats2.count == 0 {
        return (0.0, 0.0);
    }

    let mean_diff = stats1.mean - stats2.mean;
    let se1 = stats1.variance / stats1.count as f64;
    let se2 = stats2.variance / stats2.count as f64;
    let se = (se1 + se2).sqrt();

    let t_stat = if se > 0.0 { mean_diff / se } else { 0.0 };

    // Welch-Satterthwaite degrees of freedom
    let df = if se1 > 0.0 && se2 > 0.0 {
        let numerator = (se1 + se2).powi(2);
        let denom1 = se1.powi(2) / (stats1.count - 1) as f64;
        let denom2 = se2.powi(2) / (stats2.count - 1) as f64;
        numerator / (denom1 + denom2)
    } else {
        0.0
    };

    (t_stat, df)
}

/// Generate a summary report comparing two experiments
pub fn generate_report(comparison: &ExperimentComparison) -> String {
    let mut report = String::new();

    report.push_str("=== Experiment Comparison Report ===\n\n");

    report.push_str(&format!(
        "Control Strategy: {}\n",
        comparison.control_strategy
    ));
    report.push_str(&format!("Test Strategy: {}\n\n", comparison.test_strategy));

    report.push_str("--- Primary Metrics ---\n");
    report.push_str(&format!(
        "Accuracy Improvement: {:.4} ({:.2}%)\n",
        comparison.accuracy_improvement, comparison.accuracy_improvement_pct
    ));

    if let Some(delta) = comparison.convergence_epoch_delta {
        report.push_str(&format!("Convergence Epoch Delta: {} epochs\n", delta));
        if delta > 0 {
            report.push_str(&format!("  → Test converged {} epochs FASTER\n", delta));
        } else if delta < 0 {
            report.push_str(&format!(
                "  → Test converged {} epochs SLOWER\n",
                delta.abs()
            ));
        } else {
            report.push_str("  → Same convergence time\n");
        }
    } else {
        report.push_str("Convergence Epoch Delta: N/A (one or both did not converge)\n");
    }

    report.push_str("\n--- Secondary Metrics ---\n");
    report.push_str(&format!(
        "Mean Coherence Improvement: {:.4}\n",
        comparison.mean_coherence_improvement
    ));
    report.push_str(&format!(
        "Mean Energy Improvement: {:.4}\n",
        comparison.mean_energy_improvement
    ));

    report.push_str("\n--- Statistical Significance ---\n");
    report.push_str(&format!(
        "Significance Threshold: {:.4}\n",
        comparison.significance_level
    ));
    report.push_str(&format!(
        "Is Significant: {}\n",
        if comparison.is_significant {
            "YES ✓"
        } else {
            "NO"
        }
    ));

    report.push_str("\n--- Conclusion ---\n");
    if comparison.is_significant {
        report.push_str("✅ The test strategy shows statistically significant improvement.\n");
        report.push_str("   Recommendation: PROCEED with Dream Pool implementation.\n");
    } else if comparison.accuracy_improvement > 0.0 {
        report.push_str(
            "⚠️  The test strategy shows improvement, but below significance threshold.\n",
        );
        report.push_str("   Recommendation: INVESTIGATE further or tune parameters.\n");
    } else {
        report.push_str("❌ The test strategy does not show improvement.\n");
        report.push_str("   Recommendation: DEFER Dream Pool implementation.\n");
    }

    report
}

/// Analyze learning curves (accuracy over epochs)
pub fn analyze_learning_curves(
    control_metrics: &[EpochMetrics],
    test_metrics: &[EpochMetrics],
) -> LearningCurveAnalysis {
    let control_accuracies: Vec<f64> = control_metrics
        .iter()
        .map(|m| m.validation_accuracy)
        .collect();
    let test_accuracies: Vec<f64> = test_metrics.iter().map(|m| m.validation_accuracy).collect();

    let control_stats = Statistics::from_slice(&control_accuracies);
    let test_stats = Statistics::from_slice(&test_accuracies);

    // Find epoch where test first exceeds control
    let mut first_improvement_epoch = None;
    for (i, (control, test)) in control_metrics.iter().zip(test_metrics.iter()).enumerate() {
        if test.validation_accuracy > control.validation_accuracy {
            first_improvement_epoch = Some(i);
            break;
        }
    }

    LearningCurveAnalysis {
        control_stats,
        test_stats,
        first_improvement_epoch,
    }
}

/// Learning curve analysis result
#[derive(Debug, Clone, Serialize)]
pub struct LearningCurveAnalysis {
    pub control_stats: Statistics,
    pub test_stats: Statistics,
    pub first_improvement_epoch: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = Statistics::from_slice(&values);

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.count, 5);
        assert!((stats.std_dev - 1.414).abs() < 0.01);
    }

    #[test]
    fn test_welch_t_test() {
        let sample1 = vec![1.0, 2.0, 3.0];
        let sample2 = vec![2.0, 3.0, 4.0];
        let (t_stat, df) = welch_t_test(&sample1, &sample2);

        // Simple check that it computes without error
        assert!(t_stat.is_finite());
        assert!(df > 0.0);
    }
}
