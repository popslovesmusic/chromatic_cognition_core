//! Deterministic self-observation utilities for the cognition engine.
//!
//! The [`Awareness`] struct captures per-cycle statistics spanning color
//! moments, spectral descriptors, and learning dynamics (gradient energy).
//! Observations are stored in a bounded FIFO buffer to keep memory usage
//! predictable while enabling downstream forecasting.

use crate::spectral::{extract_spectral_features, WindowFunction};
use crate::tensor::ChromaticTensor;
use serde::Serialize;

/// Snapshot of core coherence signals captured during a single cycle.
#[derive(Debug, Clone, Serialize)]
pub struct Observation {
    /// Simulation step associated with the observation.
    pub step: usize,
    /// Solver coherence value reported for the current field.
    pub coherence: f32,
    /// Mean spectral entropy across RGB channels (0-1).
    pub entropy: f32,
    /// Root-mean-square gradient magnitude for the current field.
    pub grad_energy: f32,
    /// Mean RGB intensities aggregated across the tensor.
    pub mean_rgb: [f32; 3],
    /// Global color variance.
    pub variance: f32,
    /// Mean certainty value stored alongside RGB channels.
    pub mean_certainty: f32,
    /// Average low-frequency spectral energy.
    pub low_freq_energy: f32,
    /// Average mid-frequency spectral energy.
    pub mid_freq_energy: f32,
    /// Average high-frequency spectral energy.
    pub high_freq_energy: f32,
    /// Mean power spectral density across all channels.
    pub mean_psd: f32,
    /// Dominant frequency bin per RGB channel.
    pub dominant_frequencies: [usize; 3],
}

/// Rolling collector for [`Observation`] snapshots.
#[derive(Debug, Clone, Serialize)]
pub struct Awareness {
    max_history: usize,
    history: Vec<Observation>,
}

impl Awareness {
    /// Create a new awareness buffer with a bounded history length.
    pub fn new(max_history: usize) -> Self {
        Self {
            max_history: max_history.max(1),
            history: Vec::new(),
        }
    }

    /// Observe the current tensor state and return the captured statistics.
    pub fn observe(
        &mut self,
        step: usize,
        field: &ChromaticTensor,
        coherence: f32,
        gradients: Option<&[f32]>,
    ) -> Observation {
        let stats = field.statistics();
        let spectral = extract_spectral_features(field, WindowFunction::Hann);
        let grad_energy = gradients.map_or(0.0, compute_grad_energy);

        let observation = Observation {
            step,
            coherence: sanitize_scalar(coherence),
            entropy: sanitize_scalar(spectral.entropy),
            grad_energy: sanitize_scalar(grad_energy),
            mean_rgb: stats.mean_rgb,
            variance: sanitize_scalar(stats.variance),
            mean_certainty: sanitize_scalar(stats.mean_certainty),
            low_freq_energy: sanitize_scalar(spectral.low_freq_energy),
            mid_freq_energy: sanitize_scalar(spectral.mid_freq_energy),
            high_freq_energy: sanitize_scalar(spectral.high_freq_energy),
            mean_psd: sanitize_scalar(spectral.mean_psd),
            dominant_frequencies: spectral.dominant_frequencies,
        };

        if self.history.len() == self.max_history {
            self.history.remove(0);
        }
        self.history.push(observation.clone());

        observation
    }

    /// Immutable view of the stored observation history.
    pub fn history(&self) -> &[Observation] {
        &self.history
    }
}

fn compute_grad_energy(gradients: &[f32]) -> f32 {
    if gradients.is_empty() {
        return 0.0;
    }
    let sum_sq = gradients.iter().map(|value| value * value).sum::<f32>();
    (sum_sq / gradients.len() as f32).sqrt()
}

fn sanitize_scalar(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_extracts_expected_features() {
        let mut tensor = ChromaticTensor::new(2, 2, 1);
        for r in 0..2 {
            for c in 0..2 {
                tensor.certainty[[r, c, 0]] = 0.8;
                for ch in 0..3 {
                    tensor.colors[[r, c, 0, ch]] = (r + c + ch) as f32 * 0.1;
                }
            }
        }
        let gradients = vec![1.0, -1.0, 0.5, 0.25, -0.75, 0.0];
        let mut awareness = Awareness::new(4);
        let observation = awareness.observe(1, &tensor, 0.72, Some(&gradients));

        let stats = tensor.statistics();
        assert_eq!(observation.mean_rgb, stats.mean_rgb);
        assert!((observation.variance - stats.variance).abs() < 1e-6);
        assert!((observation.mean_certainty - stats.mean_certainty).abs() < 1e-6);

        let spectral = extract_spectral_features(&tensor, WindowFunction::Hann);
        assert!((observation.entropy - spectral.entropy).abs() < 1e-6);

        let expected_grad = compute_grad_energy(&gradients);
        assert!((observation.grad_energy - expected_grad).abs() < 1e-6);

        assert_eq!(awareness.history().len(), 1);
        assert_eq!(awareness.history()[0].step, 1);
    }

    #[test]
    fn history_is_bounded() {
        let tensor = ChromaticTensor::from_seed(3, 2, 2, 1);
        let gradients = vec![0.1, 0.2, 0.3];
        let mut awareness = Awareness::new(3);
        for step in 0..6 {
            awareness.observe(step, &tensor, 0.5 + step as f32 * 0.01, Some(&gradients));
        }
        assert_eq!(awareness.history().len(), 3);
        assert_eq!(awareness.history()[0].step, 3);
        assert_eq!(awareness.history()[2].step, 5);
    }
}
