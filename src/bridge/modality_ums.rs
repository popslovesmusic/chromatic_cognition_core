use std::f32::consts::{PI, TAU};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::ModalityMapper;
use crate::{
    logging,
    spectral::{accumulate::deterministic_sum, canonical_hue, SpectralTensor},
    tensor::ChromaticTensor,
};

const UMS_DIM: usize = 512;
const SPECTRAL_BAND_DIM: usize = 256;
const HSL_DIM: usize = 128;
#[derive(Debug, Clone, Serialize)]
pub struct UnifiedModalityVector {
    channels: Vec<f32>,
}

impl UnifiedModalityVector {
    pub fn components(&self) -> &[f32] {
        &self.channels
    }
}

#[derive(Debug)]
struct ChronicleNormalization {
    mu: [f32; UMS_DIM],
    sigma: [f32; UMS_DIM],
}

#[derive(Debug, Deserialize)]
struct ChronicleNormalizationRaw {
    mu: Vec<f32>,
    sigma: Vec<f32>,
}

static CHRONICLE_NORMALIZATION: OnceLock<ChronicleNormalization> = OnceLock::new();

fn chronicle_normalization() -> &'static ChronicleNormalization {
    CHRONICLE_NORMALIZATION.get_or_init(|| {
        let raw = include_str!("../../data/chronicle_ums_constants.json");
        let parsed: ChronicleNormalizationRaw =
            serde_json::from_str(raw).expect("chronicle normalization constants");
        ChronicleNormalization::from(parsed)
    })
}

pub fn encode_to_ums(mapper: &ModalityMapper, tensor: &ChromaticTensor) -> UnifiedModalityVector {
    let spectral = mapper.encode_to_spectral(tensor);
    let stats = chronicle_normalization();
    let mut channels = [0.0f32; UMS_DIM];
    channels.copy_from_slice(&stats.mu);

    let bin_count = mapper.config().spectral.fft_size / 2 + 1;
    let spectral_bins = aggregate_spectral_bins(&spectral, bin_count.max(1));
    let spectral_projection = downsample_bins(&spectral_bins);
    channels[..SPECTRAL_BAND_DIM].copy_from_slice(&spectral_projection);

    encode_hsl_block(
        &mut channels[SPECTRAL_BAND_DIM..SPECTRAL_BAND_DIM + HSL_DIM],
        tensor,
    );

    if let Err(err) = logging::log_operation("encode_to_ums", &tensor.statistics()) {
        eprintln!("failed to log encode_to_ums: {err}");
    }

    apply_normalization(&mut channels, stats);

    UnifiedModalityVector {
        channels: channels.to_vec(),
    }
}

pub fn decode_from_ums(
    mapper: &ModalityMapper,
    vector: &UnifiedModalityVector,
) -> (f32, f32, f32, usize) {
    assert_eq!(
        vector.channels.len(),
        UMS_DIM,
        "unified modality vector must contain {UMS_DIM} channels"
    );

    let stats = chronicle_normalization();
    let mut raw = [0.0f32; UMS_DIM];
    for (idx, value) in raw.iter_mut().enumerate() {
        let sigma = safe_sigma(stats.sigma[idx]);
        *value = vector.channels[idx] * sigma + stats.mu[idx];
    }

    let hue_encoded = raw[SPECTRAL_BAND_DIM];
    let hue_radians = canonical_hue((hue_encoded + 1.0) * PI);
    let saturation = raw[SPECTRAL_BAND_DIM + 1].clamp(0.0, 1.0);
    let luminance = raw[SPECTRAL_BAND_DIM + 2].clamp(0.0, 1.0);
    let category = mapper.map_hue_to_category(hue_radians);

    (hue_radians, saturation, luminance, category)
}

fn aggregate_spectral_bins(spectral: &SpectralTensor, bin_count: usize) -> Vec<f32> {
    let mut bins = vec![0.0f32; bin_count];
    let dims = spectral.components.dim();
    let f_min = spectral.f_min.max(f32::MIN_POSITIVE);
    let octave_span = spectral.octaves.max(f32::MIN_POSITIVE);
    let last_index = bin_count.saturating_sub(1);

    for row in 0..dims.0 {
        for col in 0..dims.1 {
            for layer in 0..dims.2 {
                let frequency = spectral.components[[row, col, layer, 0]].max(f_min);
                let saturation = spectral.components[[row, col, layer, 1]].clamp(0.0, 1.0);
                let value = spectral.components[[row, col, layer, 2]].clamp(0.0, 1.0);
                let energy = (saturation * value).max(0.0);

                let ratio = (frequency / f_min).max(f32::MIN_POSITIVE);
                let normalized = (ratio.log2() / octave_span).clamp(0.0, 0.999_999);
                let mut index = (normalized * last_index as f32).floor() as usize;
                if index > last_index {
                    index = last_index;
                }

                bins[index] += energy;
            }
        }
    }

    bins
}

fn downsample_bins(bins: &[f32]) -> [f32; SPECTRAL_BAND_DIM] {
    let mut projection = [0.0f32; SPECTRAL_BAND_DIM];
    if bins.is_empty() {
        return projection;
    }

    let block_size = bins.len() as f32 / SPECTRAL_BAND_DIM as f32;
    for (band, value) in projection.iter_mut().enumerate() {
        let start = (band as f32 * block_size).floor() as usize;
        let mut end = ((band as f32 + 1.0) * block_size).floor() as usize;
        if band == SPECTRAL_BAND_DIM - 1 {
            end = bins.len();
        }
        let start = start.min(bins.len());
        let end = end.max(start + 1).min(bins.len());
        let slice = &bins[start..end];
        let mean = if slice.is_empty() {
            0.0
        } else {
            deterministic_sum(slice) / slice.len() as f32
        };
        *value = mean;
    }
    projection
}

fn encode_hsl_block(block: &mut [f32], tensor: &ChromaticTensor) {
    let mean_rgb = tensor.mean_rgb();
    let (h, s, l) = rgb_to_hsl(mean_rgb);
    let hue_radians = (h * TAU).rem_euclid(TAU);
    if !block.is_empty() {
        block[0] = (hue_radians / PI) - 1.0;
    }
    if block.len() > 1 {
        block[1] = s.clamp(0.0, 1.0);
    }
    if block.len() > 2 {
        block[2] = l.clamp(0.0, 1.0);
    }
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

impl From<ChronicleNormalizationRaw> for ChronicleNormalization {
    fn from(raw: ChronicleNormalizationRaw) -> Self {
        let mu: [f32; UMS_DIM] = raw
            .mu
            .try_into()
            .expect("chronicle mu must contain 512 entries");
        let sigma: [f32; UMS_DIM] = raw
            .sigma
            .try_into()
            .expect("chronicle sigma must contain 512 entries");
        Self { mu, sigma }
    }
}

fn apply_normalization(channels: &mut [f32; UMS_DIM], stats: &ChronicleNormalization) {
    for idx in 0..UMS_DIM {
        let sigma = safe_sigma(stats.sigma[idx]);
        channels[idx] = (channels[idx] - stats.mu[idx]) / sigma;
    }
}

fn safe_sigma(value: f32) -> f32 {
    if value.abs() <= f32::EPSILON {
        1.0
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BridgeConfig;
    use std::f32::consts::TAU;

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
    fn affective_block_initializes_from_chronicle_mean() {
        let config = bridge_config();
        let mapper = ModalityMapper::new(config);
        let tensor = ChromaticTensor::from_seed(19, 4, 4, 2);

        let ums = encode_to_ums(&mapper, &tensor);
        let stats = chronicle_normalization();
        let channels = ums.components();
        let idx = UMS_DIM - 1;
        let denormalized = channels[idx] * safe_sigma(stats.sigma[idx]) + stats.mu[idx];
        assert!((denormalized - stats.mu[idx]).abs() < 1e-6);
    }

    #[test]
    fn hsl_round_trip_and_category_alignment() {
        let config = bridge_config();
        let mapper = ModalityMapper::new(config);
        let tensor = ChromaticTensor::from_seed(13, 4, 4, 2);

        let ums = encode_to_ums(&mapper, &tensor);
        let (h_rad, s, l, category) = decode_from_ums(&mapper, &ums);

        let mean_rgb = tensor.mean_rgb();
        let (expected_h_norm, expected_s, expected_l) = rgb_to_hsl(mean_rgb);
        let expected_h_rad = canonical_hue(expected_h_norm * TAU);

        assert!((h_rad - expected_h_rad).abs() < 1e-5);
        assert!((s - expected_s).abs() < 1e-5);
        assert!((l - expected_l).abs() < 1e-5);

        let expected_category = mapper.map_hue_to_category(expected_h_rad);
        assert_eq!(category, expected_category);
    }
}
