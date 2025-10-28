//! Forecast vs. actual divergence scoring utilities.
//!
//! Phase 5B introduces a dissonance metric that compares the latest
//! observation against the AR(2) forecast to quantify drift. The
//! resulting score is in the inclusive range `[0, 1]` and feeds the
//! adaptive reflection planner.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};

use serde::Serialize;

use super::awareness::Observation;
use super::predict::{Feature, PredictionSet};

const LOG_PATH: &str = "logs/meta_dissonance.jsonl";
const EPSILON: f32 = 1e-6;

/// Per-feature weights used to combine the dissonance components.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct DissonanceWeights {
    pub coherence: f32,
    pub entropy: f32,
    #[serde(rename = "energy")]
    pub energy: f32,
}

impl DissonanceWeights {
    /// Normalise weights to sum to one while keeping proportions.
    pub fn normalised(self) -> Self {
        let sum = (self.coherence + self.entropy + self.energy).abs();
        if sum <= EPSILON {
            Self {
                coherence: 1.0 / 3.0,
                entropy: 1.0 / 3.0,
                energy: 1.0 / 3.0,
            }
        } else {
            let mut coherence = (self.coherence / sum).max(0.0);
            let mut entropy = (self.entropy / sum).max(0.0);
            let mut energy = (self.energy / sum).max(0.0);
            let renorm = coherence + entropy + energy;
            if renorm <= EPSILON {
                coherence = 1.0 / 3.0;
                entropy = 1.0 / 3.0;
                energy = 1.0 / 3.0;
            } else {
                coherence /= renorm;
                entropy /= renorm;
                energy /= renorm;
            }
            Self {
                coherence,
                entropy,
                energy,
            }
        }
    }
}

/// Aggregated dissonance statistics for the current cycle.
#[derive(Debug, Clone, Serialize)]
pub struct Dissonance {
    pub coherence_delta: f32,
    pub entropy_delta: f32,
    #[serde(rename = "energy_delta")]
    pub grad_energy_delta: f32,
    pub coherence_score: f32,
    pub entropy_score: f32,
    #[serde(rename = "energy_score")]
    pub grad_energy_score: f32,
    pub score: f32,
}

impl Dissonance {
    /// Compare forecasts against the latest observation using the provided weights.
    pub fn compare(
        predictions: &PredictionSet,
        observation: &Observation,
        weights: DissonanceWeights,
    ) -> Self {
        let weights = weights.normalised();
        let predicted_coherence = forecast_value(predictions, Feature::Coherence);
        let predicted_entropy = forecast_value(predictions, Feature::Entropy);
        let predicted_energy = forecast_value(predictions, Feature::GradEnergy);

        let (coherence_delta, coherence_score) =
            component_dissonance(predicted_coherence, observation.coherence);
        let (entropy_delta, entropy_score) =
            component_dissonance(predicted_entropy, observation.entropy);
        let (grad_energy_delta, grad_energy_score) =
            component_dissonance(predicted_energy, observation.grad_energy);

        let score = (coherence_score * weights.coherence)
            + (entropy_score * weights.entropy)
            + (grad_energy_score * weights.energy);

        Self {
            coherence_delta,
            entropy_delta,
            grad_energy_delta,
            coherence_score,
            entropy_score,
            grad_energy_score,
            score: score.clamp(0.0, 1.0),
        }
    }

    /// Convenience helper to check if the score exceeds the provided threshold.
    pub fn exceeds(&self, threshold: f32) -> bool {
        self.score >= threshold
    }

    /// Write the dissonance summary to the JSON meta-log.
    pub fn log(&self, step: usize) -> io::Result<()> {
        let entry = DissonanceLogEntry {
            step,
            score: self.score,
            coherence_delta: self.coherence_delta,
            entropy_delta: self.entropy_delta,
            grad_energy_delta: self.grad_energy_delta,
        };
        append_json_line(&entry)
    }
}

#[derive(Debug, Serialize)]
struct DissonanceLogEntry {
    step: usize,
    score: f32,
    coherence_delta: f32,
    entropy_delta: f32,
    grad_energy_delta: f32,
}

fn component_dissonance(predicted: f32, actual: f32) -> (f32, f32) {
    let delta = (predicted - actual).abs();
    let scale = predicted.abs().max(actual.abs()).max(1e-3);
    let ratio = (delta / scale).max(0.0);
    let normalised = 1.0 - (-ratio).exp();
    (delta, normalised.min(1.0))
}

fn forecast_value(predictions: &PredictionSet, feature: Feature) -> f32 {
    predictions
        .values_for(feature)
        .and_then(|values| values.first().copied())
        .unwrap_or_else(|| match feature {
            Feature::Coherence => 0.0,
            Feature::Entropy => 0.0,
            Feature::GradEnergy => 0.0,
        })
}

fn append_json_line<T: Serialize>(entry: &T) -> io::Result<()> {
    if let Err(err) = fs::create_dir_all("logs") {
        return Err(err);
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_PATH)?;
    serde_json::to_writer(&mut file, entry)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
    file.write_all(b"\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::predict::{FeatureForecast, PredictionSet};

    fn make_prediction(coherence: f32, entropy: f32, grad_energy: f32) -> PredictionSet {
        PredictionSet {
            horizon: 1,
            forecasts: vec![
                FeatureForecast {
                    feature: Feature::Coherence,
                    values: vec![coherence],
                },
                FeatureForecast {
                    feature: Feature::Entropy,
                    values: vec![entropy],
                },
                FeatureForecast {
                    feature: Feature::GradEnergy,
                    values: vec![grad_energy],
                },
            ],
        }
    }

    fn make_observation(coherence: f32, entropy: f32, grad_energy: f32) -> Observation {
        Observation {
            step: 0,
            coherence,
            entropy,
            grad_energy,
            mean_rgb: [0.0, 0.0, 0.0],
            variance: 0.0,
            mean_certainty: 0.0,
            low_freq_energy: 0.0,
            mid_freq_energy: 0.0,
            high_freq_energy: 0.0,
            mean_psd: 0.0,
            dominant_frequencies: [0, 0, 0],
        }
    }

    #[test]
    fn dissonance_increases_with_noise() {
        let weights = DissonanceWeights {
            coherence: 0.5,
            entropy: 0.3,
            energy: 0.2,
        };
        let prediction = make_prediction(0.6, 0.4, 0.3);
        let base = make_observation(0.6, 0.4, 0.3);
        let mild = make_observation(0.6 + 0.05, 0.4 - 0.04, 0.3 + 0.03);
        let severe = make_observation(0.6 - 0.3, 0.4 + 0.25, 0.3 - 0.2);

        let base_score = Dissonance::compare(&prediction, &base, weights);
        let mild_score = Dissonance::compare(&prediction, &mild, weights);
        let severe_score = Dissonance::compare(&prediction, &severe, weights);

        assert!(base_score.score <= mild_score.score);
        assert!(mild_score.score <= severe_score.score);
    }

    #[test]
    fn detection_rates_meet_targets() {
        let weights = DissonanceWeights {
            coherence: 0.5,
            entropy: 0.3,
            energy: 0.2,
        };
        let prediction = make_prediction(0.55, 0.38, 0.22);
        let threshold = 0.25;
        let mut drift_detections = 0usize;
        let mut drift_cases = 0usize;
        let mut stable_false_positives = 0usize;
        let mut stable_cases = 0usize;

        for idx in 0..100 {
            let phase = (idx % 7) as f32;
            let delta = (phase - 3.0) * 0.004;
            let obs = make_observation(0.55 + delta, 0.38 - delta * 0.6, 0.22 + delta * 0.4);
            let dissonance = Dissonance::compare(&prediction, &obs, weights);
            if dissonance.exceeds(threshold) {
                stable_false_positives += 1;
            }
            stable_cases += 1;
        }

        for idx in 0..100 {
            let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
            let amplitude = if idx < 90 { 0.6 } else { 0.1 };
            let delta = sign * amplitude;
            let obs = make_observation(0.55 + delta, 0.38 - delta * 0.5, 0.22 + delta * 0.45);
            let dissonance = Dissonance::compare(&prediction, &obs, weights);
            if dissonance.exceeds(threshold) {
                drift_detections += 1;
            }
            drift_cases += 1;
        }

        let detection_rate = drift_detections as f32 / drift_cases as f32;
        let false_positive_rate = stable_false_positives as f32 / stable_cases as f32;

        assert!(detection_rate >= 0.9, "detection rate {}", detection_rate);
        assert!(
            false_positive_rate <= 0.05,
            "false positive {}",
            false_positive_rate
        );
    }
}
