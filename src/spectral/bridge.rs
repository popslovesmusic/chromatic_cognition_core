//! Hue ↔ frequency bridge for chromatic ↔ spectral tensor conversion.
//!
//! This module implements Appendix A's deterministic hue mapping:
//! - canonical hue normalization
//! - circular interpolation utilities
//! - seam blending across the red/magenta boundary
//! - reversible mapping between chromatic and spectral tensors
//!
//! The public API exposes [`SpectralTensor`] for storing frequency-domain
//! information derived from [`ChromaticTensor`]. Each tensor stores a
//! frequency channel alongside saturation and value to support a lossless
//! round-trip.

use std::f32::consts::{PI, TAU};

use ndarray::{Array3, Array4};
use serde::Serialize;

use crate::{logging, tensor::ChromaticTensor};

const DEFAULT_SEAM_EPSILON: f32 = PI * 0.05;

/// Wraps hue angles into the canonical `[0, 2π)` range.
pub fn canonical_hue(h: f32) -> f32 {
    let two_pi = TAU;
    let wrapped = h % two_pi;
    let wrapped = if wrapped < 0.0 {
        wrapped + two_pi
    } else {
        wrapped
    };
    if wrapped >= two_pi {
        wrapped - two_pi
    } else {
        wrapped
    }
}

/// Computes the circular mean of hue angles using vector addition.
pub fn circular_mean(hues: &[f32]) -> f32 {
    if hues.is_empty() {
        return 0.0;
    }
    let mut sum_sin = 0.0f32;
    let mut sum_cos = 0.0f32;
    for hue in hues {
        let canonical = canonical_hue(*hue);
        sum_sin += canonical.sin();
        sum_cos += canonical.cos();
    }
    if sum_sin.abs() < f32::EPSILON && sum_cos.abs() < f32::EPSILON {
        return 0.0;
    }
    canonical_hue(sum_sin.atan2(sum_cos))
}

fn rgb_to_hsv(rgb: [f32; 3]) -> (f32, f32, f32) {
    let r = rgb[0];
    let g = rgb[1];
    let b = rgb[2];

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let hue_sector = if delta <= f32::EPSILON {
        0.0
    } else if (max - r).abs() < f32::EPSILON {
        ((g - b) / delta).rem_euclid(6.0)
    } else if (max - g).abs() < f32::EPSILON {
        ((b - r) / delta) + 2.0
    } else {
        ((r - g) / delta) + 4.0
    };

    let hue = canonical_hue(hue_sector * (PI / 3.0));
    let saturation = if max <= f32::EPSILON {
        0.0
    } else {
        delta / max
    };
    let value = max;
    (hue, saturation, value)
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> [f32; 3] {
    let hue = canonical_hue(hue);
    let h = (hue / (PI / 3.0)).rem_euclid(6.0);
    let c = value * saturation;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = value - c;

    let (r1, g1, b1) = if h < 1.0 {
        (c, x, 0.0)
    } else if h < 2.0 {
        (x, c, 0.0)
    } else if h < 3.0 {
        (0.0, c, x)
    } else if h < 4.0 {
        (0.0, x, c)
    } else if h < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    [
        (r1 + m).clamp(0.0, 1.0),
        (g1 + m).clamp(0.0, 1.0),
        (b1 + m).clamp(0.0, 1.0),
    ]
}

#[derive(Clone, Debug)]
struct HueFrequencyBridge {
    f_min: f32,
    octaves: f32,
    seam_epsilon: f32,
}

impl HueFrequencyBridge {
    fn new(f_min: f32, octaves: f32, seam_epsilon: f32) -> Self {
        assert!(f_min.is_finite() && f_min > 0.0, "f_min must be positive");
        assert!(
            octaves.is_finite() && octaves > 0.0,
            "octaves must be positive"
        );
        assert!(
            seam_epsilon.is_finite() && seam_epsilon > 0.0 && seam_epsilon < TAU,
            "seam epsilon must be within (0, 2π)"
        );
        Self {
            f_min,
            octaves,
            seam_epsilon,
        }
    }

    fn raw_frequency(&self, hue: f32) -> f32 {
        self.f_min * (2.0f32).powf((hue / TAU) * self.octaves)
    }

    fn blend_low(&self, hue: f32) -> f32 {
        let base = self.raw_frequency(hue);
        let alt = self.raw_frequency(hue + TAU);
        let alpha = 1.0 - (hue / self.seam_epsilon);
        (alpha * base) + ((1.0 - alpha) * alt)
    }

    fn blend_high(&self, hue: f32) -> f32 {
        let base = self.raw_frequency(hue);
        let alt = self.raw_frequency(hue - TAU);
        let delta = TAU - hue;
        let alpha = 1.0 - (delta / self.seam_epsilon);
        (alpha * base) + ((1.0 - alpha) * alt)
    }

    fn hue_to_frequency(&self, hue: f32) -> (f32, f32) {
        let hue = canonical_hue(hue);
        if hue < self.seam_epsilon {
            let frequency = self.blend_low(hue);
            let alpha = 1.0 - (hue / self.seam_epsilon);
            return (frequency, alpha);
        }
        if hue > TAU - self.seam_epsilon {
            let frequency = self.blend_high(hue);
            let delta = TAU - hue;
            let alpha = 1.0 - (delta / self.seam_epsilon);
            return (frequency, -alpha);
        }
        (self.raw_frequency(hue), 0.0)
    }

    fn frequency_to_hue(&self, frequency: f32, seam_flag: f32) -> f32 {
        if seam_flag > 0.0 {
            let alpha = seam_flag.clamp(0.0, 1.0);
            return canonical_hue((1.0 - alpha) * self.seam_epsilon);
        }
        if seam_flag < 0.0 {
            let alpha = (-seam_flag).clamp(0.0, 1.0);
            return canonical_hue(TAU - (1.0 - alpha) * self.seam_epsilon);
        }
        let safe_frequency = frequency.max(f32::MIN_POSITIVE);
        let ratio = safe_frequency / self.f_min;
        let raw = (ratio.log2() / self.octaves) * TAU;
        canonical_hue(raw)
    }
}

/// Spectral tensor storing frequency, saturation, value, and seam channels.
#[derive(Clone, Debug, Serialize)]
pub struct SpectralTensor {
    /// Spectral components `[rows, cols, layers, (frequency, saturation, value, seam_flag)]`.
    pub components: Array4<f32>,
    /// Certainty weights shared with the originating chromatic tensor.
    pub certainty: Array3<f32>,
    /// Minimum frequency in Hertz.
    pub f_min: f32,
    /// Number of octaves covered by the hue spectrum.
    pub octaves: f32,
    /// Seam blending width.
    pub seam_epsilon: f32,
}

impl SpectralTensor {
    /// Create a new spectral tensor with zeroed components.
    pub fn new(rows: usize, cols: usize, layers: usize, f_min: f32, octaves: f32) -> Self {
        Self::with_epsilon(rows, cols, layers, f_min, octaves, DEFAULT_SEAM_EPSILON)
    }

    /// Create a new spectral tensor with a custom seam epsilon.
    pub fn with_epsilon(
        rows: usize,
        cols: usize,
        layers: usize,
        f_min: f32,
        octaves: f32,
        seam_epsilon: f32,
    ) -> Self {
        SpectralTensor {
            components: Array4::zeros((rows, cols, layers, 4)),
            certainty: Array3::zeros((rows, cols, layers)),
            f_min,
            octaves,
            seam_epsilon,
        }
    }

    /// Convert a chromatic tensor into the spectral domain.
    pub fn from_chromatic(tensor: &ChromaticTensor, f_min: f32, octaves: f32) -> Self {
        Self::from_chromatic_with_epsilon(tensor, f_min, octaves, DEFAULT_SEAM_EPSILON)
    }

    /// Convert a chromatic tensor using a custom seam epsilon.
    pub fn from_chromatic_with_epsilon(
        tensor: &ChromaticTensor,
        f_min: f32,
        octaves: f32,
        seam_epsilon: f32,
    ) -> Self {
        let mapper = HueFrequencyBridge::new(f_min, octaves, seam_epsilon);
        let (rows, cols, layers, _) = tensor.shape();
        let mut components = Array4::zeros((rows, cols, layers, 4));
        for row in 0..rows {
            for col in 0..cols {
                for layer in 0..layers {
                    let rgb = tensor.get_rgb(row, col, layer);
                    let (hue, saturation, value) = rgb_to_hsv(rgb);
                    let (frequency, seam_flag) = mapper.hue_to_frequency(hue);
                    components[[row, col, layer, 0]] = frequency;
                    components[[row, col, layer, 1]] = saturation;
                    components[[row, col, layer, 2]] = value;
                    components[[row, col, layer, 3]] = seam_flag;
                }
            }
        }
        if let Err(err) = logging::log_operation("chromatic_to_spectral", &tensor.statistics()) {
            eprintln!("failed to log chromatic_to_spectral: {err}");
        }
        SpectralTensor {
            components,
            certainty: tensor.certainty.clone(),
            f_min,
            octaves,
            seam_epsilon,
        }
    }

    /// Convert the spectral tensor back into chromatic space.
    pub fn to_chromatic(&self) -> ChromaticTensor {
        let mapper = HueFrequencyBridge::new(self.f_min, self.octaves, self.seam_epsilon);
        let (rows, cols, layers, _) = self.components.dim();
        let mut colors = Array4::zeros((rows, cols, layers, 3));
        for row in 0..rows {
            for col in 0..cols {
                for layer in 0..layers {
                    let frequency = self.components[[row, col, layer, 0]];
                    let saturation = self.components[[row, col, layer, 1]];
                    let value = self.components[[row, col, layer, 2]];
                    let seam_flag = self.components[[row, col, layer, 3]];
                    let hue = mapper.frequency_to_hue(frequency, seam_flag);
                    let rgb = hsv_to_rgb(hue, saturation, value);
                    colors[[row, col, layer, 0]] = rgb[0];
                    colors[[row, col, layer, 1]] = rgb[1];
                    colors[[row, col, layer, 2]] = rgb[2];
                }
            }
        }
        let chromatic = ChromaticTensor::from_arrays(colors, self.certainty.clone());
        if let Err(err) = logging::log_operation("spectral_to_chromatic", &chromatic.statistics()) {
            eprintln!("failed to log spectral_to_chromatic: {err}");
        }
        chromatic
    }

    /// Get tensor dimensions.
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        self.components.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_hue() {
        let base = TAU * 1.75;
        let normalized = canonical_hue(base);
        assert!(normalized >= 0.0 && normalized < TAU);
        let negative = canonical_hue(-PI / 2.0);
        assert!((negative - (1.5 * PI)).abs() < 1e-6);
    }

    #[test]
    fn circular_mean_handles_wrap() {
        let hues = [0.0, TAU - 0.1];
        let mean = circular_mean(&hues);
        assert!(mean < 0.1 || mean > TAU - 0.1);
    }

    #[test]
    fn hue_frequency_round_trip_near_seam() {
        let mapper = HueFrequencyBridge::new(220.0, 6.0, DEFAULT_SEAM_EPSILON);
        let original = 0.01;
        let (frequency, seam_flag) = mapper.hue_to_frequency(original);
        let reconstructed = mapper.frequency_to_hue(frequency, seam_flag);
        assert!((original - reconstructed).abs() < 1e-6);
    }

    #[test]
    fn chromatic_spectral_round_trip() {
        let tensor = ChromaticTensor::from_seed(42, 4, 4, 2);
        let spectral = SpectralTensor::from_chromatic(&tensor, 110.0, 5.0);
        let reconstructed = spectral.to_chromatic();
        let (rows, cols, layers, _) = tensor.shape();
        for row in 0..rows {
            for col in 0..cols {
                for layer in 0..layers {
                    let original = tensor.get_rgb(row, col, layer);
                    let roundtrip = reconstructed.get_rgb(row, col, layer);
                    for channel in 0..3 {
                        assert!(
                            (original[channel] - roundtrip[channel]).abs() < 1e-5,
                            "channel mismatch"
                        );
                    }
                }
            }
        }
    }
}
