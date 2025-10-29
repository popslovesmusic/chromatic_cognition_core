use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::ModalityMapper;
use crate::{logging, spectral::SpectralTensor, tensor::ChromaticTensor};

const UMS_DIM: usize = 512;
const SPECTRAL_BAND_DIM: usize = 256;
const HSL_DIM: usize = 128;
const CATEGORY_COUNT: usize = 12;

#[derive(Debug, Clone, Serialize)]
pub struct UnifiedModalityVector {
    channels: Vec<f32>,
}

impl UnifiedModalityVector {
    pub fn components(&self) -> &[f32] {
        &self.channels
    }
}

#[derive(Debug, Deserialize)]
struct ChronicleNormalization {
    mu: f32,
    sigma: f32,
}

static CHRONICLE_NORMALIZATION: OnceLock<ChronicleNormalization> = OnceLock::new();

fn chronicle_normalization() -> &'static ChronicleNormalization {
    CHRONICLE_NORMALIZATION.get_or_init(|| {
        let raw = include_str!("../../data/chronicle_ums_constants.json");
        serde_json::from_str(raw).expect("chronicle normalization constants")
    })
}

pub fn encode_to_ums(mapper: &ModalityMapper, tensor: &ChromaticTensor) -> UnifiedModalityVector {
    let spectral = mapper.encode_to_spectral(tensor);
    let categories = compute_category_distribution(
        &spectral,
        mapper.config().spectral.categorical_count,
    );

    let mut channels = [0.0f32; UMS_DIM];
    let spectral_projection = project_spectral(&categories);
    channels[..SPECTRAL_BAND_DIM].copy_from_slice(&spectral_projection);

    let hsl_projection = project_hsl(tensor);
    let hsl_start = SPECTRAL_BAND_DIM;
    let hsl_end = hsl_start + HSL_DIM;
    channels[hsl_start..hsl_end].copy_from_slice(&hsl_projection);

    let stats = chronicle_normalization();
    let sigma = if stats.sigma <= f32::EPSILON { 1.0 } else { stats.sigma };

    for value in channels[hsl_end..].iter_mut() {
        *value = stats.mu;
    }

    if let Err(err) = logging::log_operation("encode_to_ums", &tensor.statistics()) {
        eprintln!("failed to log encode_to_ums: {err}");
    }

    for value in channels.iter_mut() {
        *value = (*value - stats.mu) / sigma;
    }

    UnifiedModalityVector {
        channels: channels.to_vec(),
    }
}

pub fn decode_from_ums(vector: &UnifiedModalityVector) -> (f32, f32, f32) {
    let stats = chronicle_normalization();
    let sigma = if stats.sigma <= f32::EPSILON { 1.0 } else { stats.sigma };

    let hsl_start = SPECTRAL_BAND_DIM;
    let hsl_end = hsl_start + HSL_DIM;
    let mut accum = [0.0f32; 3];
    let mut counts = [0usize; 3];

    for (idx, &value) in vector.channels[hsl_start..hsl_end].iter().enumerate() {
        let raw = value * sigma + stats.mu;
        let slot = idx % 3;
        accum[slot] += raw;
        counts[slot] += 1;
    }

    let mut result = [0.0f32; 3];
    for (idx, total) in accum.iter().enumerate() {
        let count = counts[idx].max(1) as f32;
        result[idx] = (total / count).clamp(0.0, 1.0);
    }
    (result[0], result[1], result[2])
}

fn compute_category_distribution(
    spectral: &SpectralTensor,
    configured_count: usize,
) -> [f32; CATEGORY_COUNT] {
    let active = CATEGORY_COUNT.min(configured_count.max(1));
    let dims = spectral.components.dim();
    let mut sums = [0.0f32; CATEGORY_COUNT];
    let mut total_energy = 0.0f32;
    let f_min = spectral.f_min.max(f32::MIN_POSITIVE);
    let octave_span = spectral.octaves.max(f32::MIN_POSITIVE);

    for row in 0..dims.0 {
        for col in 0..dims.1 {
            for layer in 0..dims.2 {
                let frequency = spectral.components[[row, col, layer, 0]].max(f_min);
                let saturation = spectral.components[[row, col, layer, 1]].clamp(0.0, 1.0);
                let value = spectral.components[[row, col, layer, 2]].clamp(0.0, 1.0);
                let energy = (saturation * value).max(0.0);

                let ratio = (frequency / f_min).max(f32::MIN_POSITIVE);
                let normalized = (ratio.log2() / octave_span).clamp(0.0, 0.999_999);
                let mut index = (normalized * active as f32).floor() as usize;
                if index >= active {
                    index = active - 1;
                }

                sums[index] += energy;
                total_energy += energy;
            }
        }
    }

    if total_energy <= f32::EPSILON {
        let uniform = 1.0f32 / active as f32;
        for idx in 0..active {
            sums[idx] = uniform;
        }
    } else {
        for idx in 0..active {
            sums[idx] /= total_energy;
        }
    }

    sums
}

fn project_spectral(categories: &[f32; CATEGORY_COUNT]) -> [f32; SPECTRAL_BAND_DIM] {
    let mut projection = [0.0f32; SPECTRAL_BAND_DIM];
    for (idx, value) in projection.iter_mut().enumerate() {
        let mapped = ((idx as f32 / SPECTRAL_BAND_DIM as f32) * CATEGORY_COUNT as f32)
            .floor() as usize;
        let mapped = mapped.min(CATEGORY_COUNT - 1);
        *value = categories[mapped];
    }
    projection
}

fn project_hsl(tensor: &ChromaticTensor) -> [f32; HSL_DIM] {
    let mean_rgb = tensor.mean_rgb();
    let (h, s, l) = rgb_to_hsl(mean_rgb);
    let features = [h, s, l];
    let mut projection = [0.0f32; HSL_DIM];
    for (idx, value) in projection.iter_mut().enumerate() {
        *value = features[idx % features.len()];
    }
    projection
}

fn rgb_to_hsl(rgb: [f32; 3]) -> (f32, f32, f32) {
    let r = rgb[0].clamp(0.0, 1.0);
    let g = rgb[1].clamp(0.0, 1.0);
    let b = rgb[2].clamp(0.0, 1.0);

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    let delta = max - min;

    if delta <= f32::EPSILON {
        return (0.0, 0.0, l);
    }

    let s = if l <= 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };

    let hue_sector = if (max - r).abs() < f32::EPSILON {
        ((g - b) / delta).rem_euclid(6.0)
    } else if (max - g).abs() < f32::EPSILON {
        ((b - r) / delta) + 2.0
    } else {
        ((r - g) / delta) + 4.0
    };

    let h = (hue_sector / 6.0).rem_euclid(1.0);
    (h, s.clamp(0.0, 1.0), l.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BridgeConfig;

    fn bridge_config() -> BridgeConfig {
        BridgeConfig::from_str(
            r#"
            [bridge]
            f_min = 110.0
            octaves = 7.0
            gamma = 1.0
            sample_rate = 44100

            [bridge.spectral]
            fft_size = 1024
            accum_format = "Q16.48"
            reduction_mode = "pairwise_neumaier"
            categorical_count = 12

            [bridge.reversibility]
            delta_e_tolerance = 0.001
            "#,
        )
        .expect("valid bridge config")
    }

    #[test]
    fn spectral_categories_normalize() {
        let config = bridge_config();
        let mapper = ModalityMapper::new(config.clone());
        let tensor = ChromaticTensor::from_seed(7, 4, 4, 2);
        let spectral = mapper.encode_to_spectral(&tensor);
        let categories = compute_category_distribution(&spectral, config.spectral.categorical_count);
        let sum: f32 = categories.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn hsl_round_trip_matches_projection() {
        let config = bridge_config();
        let mapper = ModalityMapper::new(config);
        let tensor = ChromaticTensor::from_seed(13, 4, 4, 2);

        let ums = encode_to_ums(&mapper, &tensor);
        let (h, s, l) = decode_from_ums(&ums);

        let mean_rgb = tensor.mean_rgb();
        let (expected_h, expected_s, expected_l) = rgb_to_hsl(mean_rgb);

        assert!((h - expected_h).abs() < 1e-5);
        assert!((s - expected_s).abs() < 1e-5);
        assert!((l - expected_l).abs() < 1e-5);

        let stats = chronicle_normalization();
        let sigma = if stats.sigma <= f32::EPSILON { 1.0 } else { stats.sigma };
        let last = ums.components()[UMS_DIM - 1] * sigma + stats.mu;
        assert!((last - stats.mu).abs() < 1e-6);
    }
}
