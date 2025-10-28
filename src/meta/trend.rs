//! Long-term trend analysis utilities for the meta layer.
//!
//! This module performs linear regression over recent cycle history to
//! estimate directional drift alongside FFT-based oscillation detection.

use chrono::{DateTime, Utc};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use serde::Serialize;
use uuid::Uuid;

use crate::config::Phase6BConfig;

/// Chronicle record captured for each training cycle.
#[derive(Debug, Clone, Serialize)]
pub struct CycleRecord {
    pub id: u64,
    pub timestamp: DateTime<Utc>,
    pub coherence: f32,
    pub entropy: f32,
    pub grad_energy: f32,
    pub loss: f32,
    pub val_accuracy: f32,
    pub dream_seed: Option<Uuid>,
    pub meta_score: f32,
}

/// Aggregated trend statistics derived from the chronicle history.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct TrendModel {
    pub slope_coherence: f32,
    pub slope_entropy: f32,
    pub slope_loss: f32,
    pub oscillation_period: f32,
}

impl TrendModel {
    fn zeroed() -> Self {
        Self {
            slope_coherence: 0.0,
            slope_entropy: 0.0,
            slope_loss: 0.0,
            oscillation_period: 0.0,
        }
    }
}

/// Fit regression slopes and detect oscillations for the supplied history.
pub fn fit_trend(history: &[CycleRecord]) -> TrendModel {
    if history.is_empty() {
        return TrendModel::zeroed();
    }

    let config = Phase6BConfig::default();
    let window = config.trend_window.max(2);
    let start = history.len().saturating_sub(window);
    let slice = &history[start..];

    if slice.len() < 2 {
        return TrendModel::zeroed();
    }

    let coherence: Vec<f32> = slice.iter().map(|record| record.coherence).collect();
    let entropy: Vec<f32> = slice.iter().map(|record| record.entropy).collect();
    let loss: Vec<f32> = slice.iter().map(|record| record.loss).collect();

    let slope_coherence = linear_slope(&coherence);
    let slope_entropy = linear_slope(&entropy);
    let slope_loss = linear_slope(&loss);
    let oscillation_period = detect_oscillation(&coherence, config.oscillation_limit);

    TrendModel {
        slope_coherence,
        slope_entropy,
        slope_loss,
        oscillation_period,
    }
}

/// Determine whether the latest record is anomalous relative to the trend model.
pub fn detect_anomaly(trend: &TrendModel, latest: &CycleRecord) -> bool {
    let config = Phase6BConfig::default();
    let drift_limit = config.trend_drift_limit.max(1e-6);

    let coherence_change = relative_change(trend.slope_coherence, latest.coherence);
    let entropy_change = relative_change(trend.slope_entropy, latest.entropy);
    let loss_change = relative_change(trend.slope_loss, latest.loss);

    let drift_detected = coherence_change.abs() > drift_limit
        || entropy_change.abs() > drift_limit
        || loss_change.abs() > drift_limit;
    let oscillation_detected = trend.oscillation_period > 0.0;

    drift_detected || oscillation_detected
}

fn linear_slope(series: &[f32]) -> f32 {
    let len = series.len();
    if len < 2 {
        return 0.0;
    }

    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_xy = 0.0f32;
    let mut sum_x2 = 0.0f32;

    for (idx, &value) in series.iter().enumerate() {
        let x = idx as f32;
        sum_x += x;
        sum_y += value;
        sum_xy += x * value;
        sum_x2 += x * x;
    }

    let n = len as f32;
    let denominator = n * sum_x2 - sum_x * sum_x;
    if denominator.abs() < f32::EPSILON {
        return 0.0;
    }

    (n * sum_xy - sum_x * sum_y) / denominator
}

fn detect_oscillation(series: &[f32], prominence_limit: f32) -> f32 {
    let len = series.len();
    if len < 4 {
        return 0.0;
    }

    let mean = series.iter().copied().sum::<f32>() / len as f32;
    let mut buffer: Vec<Complex32> = series
        .iter()
        .map(|&value| Complex32::new(value - mean, 0.0))
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(len);
    fft.process(&mut buffer);

    let mut total = 0.0f32;
    let mut strongest = 0.0f32;
    let mut strongest_index = 0usize;

    let upper = len / 2;
    for idx in 1..=upper {
        let magnitude = buffer[idx].norm();
        total += magnitude;
        if magnitude > strongest {
            strongest = magnitude;
            strongest_index = idx;
        }
    }

    if strongest_index == 0 || total <= f32::EPSILON {
        return 0.0;
    }

    let prominence = strongest / total;
    if prominence < prominence_limit {
        return 0.0;
    }

    (len as f32) / (strongest_index as f32)
}

fn relative_change(slope: f32, latest: f32) -> f32 {
    let baseline = latest.abs().max(1e-3);
    slope / baseline
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn record(step: i64, coherence: f32, entropy: f32, loss: f32) -> CycleRecord {
        CycleRecord {
            id: step as u64,
            timestamp: Utc.timestamp_opt(step, 0).unwrap(),
            coherence,
            entropy,
            grad_energy: 0.0,
            loss,
            val_accuracy: 0.9,
            dream_seed: Some(Uuid::nil()),
            meta_score: 0.5,
        }
    }

    #[test]
    fn regression_slopes_match_manual_estimate() {
        let mut history = Vec::new();
        for step in 0..20 {
            let coherence = 0.6 + 0.02 * step as f32;
            let entropy = 1.2 - 0.03 * step as f32;
            let loss = 0.5 + 0.01 * step as f32;
            history.push(record(step, coherence, entropy, loss));
        }

        let trend = fit_trend(&history);
        assert!((trend.slope_coherence - 0.02).abs() < 1e-5);
        assert!((trend.slope_entropy + 0.03).abs() < 1e-5);
        assert!((trend.slope_loss - 0.01).abs() < 1e-5);
    }

    #[test]
    fn oscillation_detection_identifies_period() {
        let mut history = Vec::new();
        for step in 0..20 {
            let angle = (step as f32) * std::f32::consts::TAU / 5.0;
            let coherence = 0.5 + 0.1 * angle.sin();
            history.push(record(step, coherence, 1.0, 0.2));
        }

        let trend = fit_trend(&history);
        assert!(trend.oscillation_period > 0.0);
        assert!((trend.oscillation_period - 5.0).abs() < 0.6);
    }

    #[test]
    fn detect_anomaly_flags_drift() {
        let latest = record(99, 1.0, 0.5, 0.4);
        let trend = TrendModel {
            slope_coherence: -0.05,
            slope_entropy: 0.01,
            slope_loss: 0.0,
            oscillation_period: 0.0,
        };

        assert!(detect_anomaly(&trend, &latest));
    }

    #[test]
    fn detect_anomaly_flags_oscillation() {
        let latest = record(42, 0.8, 1.1, 0.3);
        let trend = TrendModel {
            slope_coherence: 0.0,
            slope_entropy: 0.0,
            slope_loss: 0.0,
            oscillation_period: 4.0,
        };

        assert!(detect_anomaly(&trend, &latest));
    }

    #[test]
    fn detect_anomaly_passes_stable_signal() {
        let latest = record(10, 0.9, 0.8, 0.2);
        let trend = TrendModel {
            slope_coherence: 0.005,
            slope_entropy: -0.004,
            slope_loss: 0.002,
            oscillation_period: 0.0,
        };

        assert!(!detect_anomaly(&trend, &latest));
    }
}
