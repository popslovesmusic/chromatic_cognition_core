//! Short-term forecasting utilities for awareness signals.
//!
//! Implements an autoregressive (AR(2)) predictor for coherence-related
//! features captured by [`Awareness`](super::awareness).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::str::FromStr;

use serde::Serialize;

use super::awareness::Observation;

const DEFAULT_MAX_HISTORY: usize = 256;
const DEFAULT_REGULARIZATION: f32 = 1e-3;
const COEFF_LIMIT: f32 = 1.2;
const DAMPING_FACTOR: f32 = 0.95;

/// Forecastable metrics tracked by the predictor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Feature {
    Coherence,
    Entropy,
    GradEnergy,
}

impl Feature {
    pub fn as_str(&self) -> &'static str {
        match self {
            Feature::Coherence => "coherence",
            Feature::Entropy => "entropy",
            Feature::GradEnergy => "grad_energy",
        }
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for Feature {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "coherence" => Ok(Feature::Coherence),
            "entropy" => Ok(Feature::Entropy),
            "grad_energy" => Ok(Feature::GradEnergy),
            _ => Err(format!("unsupported feature '{}'", s)),
        }
    }
}

/// Per-feature forecast for a fixed horizon.
#[derive(Debug, Clone, Serialize)]
pub struct FeatureForecast {
    pub feature: Feature,
    pub values: Vec<f32>,
}

/// Bundle of forecasts returned by [`Predictor::predict`].
#[derive(Debug, Clone, Serialize)]
pub struct PredictionSet {
    pub horizon: usize,
    pub forecasts: Vec<FeatureForecast>,
}

impl PredictionSet {
    /// Convenience accessor for a particular feature's forecast values.
    pub fn values_for(&self, feature: Feature) -> Option<&[f32]> {
        self.forecasts
            .iter()
            .find(|forecast| forecast.feature == feature)
            .map(|forecast| forecast.values.as_slice())
    }
}

/// Deterministic AR(2) predictor for awareness features.
#[derive(Debug, Clone, Serialize)]
pub struct Predictor {
    horizon: usize,
    features: Vec<Feature>,
    history: HashMap<Feature, Vec<f32>>,
    max_history: usize,
    regularization: f32,
}

impl Predictor {
    /// Create a predictor with the requested horizon and feature set.
    pub fn new(horizon: usize, features: Vec<Feature>) -> Self {
        let mut seen = HashSet::new();
        let mut unique = Vec::new();
        for feature in features {
            if seen.insert(feature) {
                unique.push(feature);
            }
        }
        let history = unique
            .iter()
            .map(|feature| (*feature, Vec::new()))
            .collect::<HashMap<_, _>>();

        Self {
            horizon: horizon.max(1),
            features: unique,
            history,
            max_history: DEFAULT_MAX_HISTORY,
            regularization: DEFAULT_REGULARIZATION,
        }
    }

    /// Update the predictor with a new observation.
    pub fn update(&mut self, observation: &Observation) {
        for feature in &self.features {
            let series = self.history.entry(*feature).or_default();
            series.push(select_feature(observation, *feature));
            if series.len() > self.max_history {
                series.remove(0);
            }
        }
    }

    /// Generate AR(2)-based forecasts for the configured horizon.
    pub fn predict(&self) -> PredictionSet {
        let forecasts = self
            .features
            .iter()
            .map(|feature| {
                let series = self.history.get(feature).map_or(&[][..], |v| v.as_slice());
                let values = self.forecast_feature(*feature, series);
                FeatureForecast {
                    feature: *feature,
                    values,
                }
            })
            .collect();

        PredictionSet {
            horizon: self.horizon,
            forecasts,
        }
    }

    fn forecast_feature(&self, feature: Feature, series: &[f32]) -> Vec<f32> {
        if series.is_empty() {
            return vec![0.0; self.horizon];
        }
        if series.len() == 1 {
            return vec![series[0]; self.horizon];
        }

        let (a1, a2) = estimate_ar2(series, self.regularization);
        let mut prev1 = *series.last().unwrap();
        let mut prev2 = series[series.len() - 2];
        let (min_val, max_val) = bounds(series);
        let range = (max_val - min_val).abs();
        let margin = (range * 0.25).max(1e-3);
        let lower = min_val - margin;
        let upper = max_val + margin;
        let mean = series.iter().copied().sum::<f32>() / series.len() as f32;

        let mut values = Vec::with_capacity(self.horizon);
        for _ in 0..self.horizon {
            let mut next = a1 * prev1 + a2 * prev2;
            if !next.is_finite() {
                next = prev1;
            }
            next = DAMPING_FACTOR * next + (1.0 - DAMPING_FACTOR) * mean;
            if matches!(feature, Feature::GradEnergy) {
                next = next.max(0.0);
            }
            next = next.clamp(lower, upper);
            values.push(next);
            prev2 = prev1;
            prev1 = next;
        }

        values
    }
}

fn select_feature(observation: &Observation, feature: Feature) -> f32 {
    match feature {
        Feature::Coherence => observation.coherence,
        Feature::Entropy => observation.entropy,
        Feature::GradEnergy => observation.grad_energy,
    }
}

fn estimate_ar2(series: &[f32], lambda: f32) -> (f32, f32) {
    if series.len() < 3 {
        return (1.0, 0.0);
    }

    let mut s11 = 0.0f32;
    let mut s22 = 0.0f32;
    let mut s12 = 0.0f32;
    let mut sy1 = 0.0f32;
    let mut sy2 = 0.0f32;

    for t in 2..series.len() {
        let x1 = series[t - 1];
        let x2 = series[t - 2];
        let y = series[t];
        s11 += x1 * x1;
        s22 += x2 * x2;
        s12 += x1 * x2;
        sy1 += x1 * y;
        sy2 += x2 * y;
    }

    s11 += lambda;
    s22 += lambda;

    let det = s11 * s22 - s12 * s12;
    if det.abs() < 1e-6 {
        return (1.0, 0.0);
    }

    let mut a1 = (s22 * sy1 - s12 * sy2) / det;
    let mut a2 = (s11 * sy2 - s12 * sy1) / det;

    let magnitude = a1.abs() + a2.abs();
    if magnitude > COEFF_LIMIT {
        let scale = COEFF_LIMIT / magnitude;
        a1 *= scale;
        a2 *= scale;
    }

    (
        a1.clamp(-COEFF_LIMIT, COEFF_LIMIT),
        a2.clamp(-COEFF_LIMIT, COEFF_LIMIT),
    )
}

fn bounds(series: &[f32]) -> (f32, f32) {
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &value in series {
        if value < min_val {
            min_val = value;
        }
        if value > max_val {
            max_val = value;
        }
    }
    if !min_val.is_finite() || !max_val.is_finite() {
        (0.0, 0.0)
    } else {
        (min_val, max_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_observation(step: usize, coherence: f32) -> Observation {
        Observation {
            step,
            coherence,
            entropy: 0.4 + 0.001 * step as f32,
            grad_energy: 0.2 + 0.002 * step as f32,
            mean_rgb: [0.1, 0.2, 0.3],
            variance: 0.01,
            mean_certainty: 0.8,
            low_freq_energy: 0.5,
            mid_freq_energy: 0.25,
            high_freq_energy: 0.15,
            mean_psd: 0.2,
            dominant_frequencies: [1, 2, 3],
        }
    }

    #[test]
    fn predictor_generates_bounded_forecasts() {
        let mut predictor = Predictor::new(
            3,
            vec![Feature::Coherence, Feature::Entropy, Feature::GradEnergy],
        );
        for step in 0..32 {
            let coherence = (step as f32 * 0.05).sin();
            let mut obs = build_observation(step, coherence);
            obs.entropy += (step as f32 * 0.03).cos() * 0.05;
            obs.grad_energy = 0.1 + (step as f32 * 0.07).sin().abs();
            predictor.update(&obs);
        }

        let predictions = predictor.predict();
        for forecast in &predictions.forecasts {
            assert_eq!(forecast.values.len(), predictions.horizon);
            for value in &forecast.values {
                assert!(value.is_finite());
            }
        }
    }

    #[test]
    fn predictor_is_deterministic() {
        let steps = 20;
        let features = vec![Feature::Coherence, Feature::Entropy];
        let mut predictor_a = Predictor::new(2, features.clone());
        let mut predictor_b = Predictor::new(2, features.clone());

        for step in 0..steps {
            let obs = build_observation(step, 0.5 + 0.01 * step as f32);
            predictor_a.update(&obs);
            predictor_b.update(&obs);
        }

        let forecast_a = predictor_a.predict();
        let forecast_b = predictor_b.predict();

        for feature in features {
            let values_a = forecast_a.values_for(feature).unwrap();
            let values_b = forecast_b.values_for(feature).unwrap();
            assert_eq!(values_a, values_b);
        }
    }

    #[test]
    fn predictor_hits_correlation_target() {
        let mut predictor = Predictor::new(2, vec![Feature::Coherence]);
        let mut series = vec![0.3f32, 0.1f32];
        for idx in 2..120 {
            let next = 0.72 * series[idx - 1] - 0.18 * series[idx - 2];
            series.push(next);
        }

        let mut predicted = Vec::new();
        let mut actual = Vec::new();
        for step in 0..series.len() {
            if step >= 2 {
                let forecast = predictor.predict();
                if let Some(values) = forecast.values_for(Feature::Coherence) {
                    predicted.push(values[0]);
                    actual.push(series[step]);
                }
            }
            let obs = build_observation(step, series[step]);
            predictor.update(&obs);
        }

        let correlation = pearson(&predicted, &actual);
        assert!(correlation > 0.8, "correlation {}", correlation);
    }

    fn pearson(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        let mean_x = x.iter().copied().sum::<f32>() / n as f32;
        let mean_y = y.iter().copied().sum::<f32>() / n as f32;

        let mut num = 0.0f32;
        let mut denom_x = 0.0f32;
        let mut denom_y = 0.0f32;
        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            num += dx * dy;
            denom_x += dx * dx;
            denom_y += dy * dy;
        }
        if denom_x <= 1e-12 || denom_y <= 1e-12 {
            0.0
        } else {
            num / (denom_x.sqrt() * denom_y.sqrt())
        }
    }
}
