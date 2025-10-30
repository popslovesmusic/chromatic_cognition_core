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

#[derive(Clone, Copy, Debug)]
struct HslSample {
    hue: f32,
    saturation: f32,
    luminance: f32,
    layer: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct HslStats {
    sin_sum: f32,
    cos_sum: f32,
    weight_sum: f32,
    saturation_sum: f32,
    luminance_sum: f32,
    count: usize,
}

impl HslStats {
    fn add(&mut self, sample: &HslSample) {
        let weight = sample.saturation.max(0.0);
        if weight > 0.0 {
            self.sin_sum += sample.hue.sin() * weight;
            self.cos_sum += sample.hue.cos() * weight;
            self.weight_sum += weight;
        }
        self.saturation_sum += sample.saturation;
        self.luminance_sum += sample.luminance;
        self.count += 1;
    }

    fn means(&self) -> Option<(f32, f32, f32)> {
        if self.count == 0 {
            return None;
        }
        let hue = if self.weight_sum <= f32::EPSILON {
            0.0
        } else {
            (self.sin_sum / self.weight_sum)
                .atan2(self.cos_sum / self.weight_sum)
                .rem_euclid(TAU)
        };
        let inv = 1.0 / self.count as f32;
        let saturation = (self.saturation_sum * inv).clamp(0.0, 1.0);
        let luminance = (self.luminance_sum * inv).clamp(0.0, 1.0);
        Some((hue, saturation, luminance))
    }
}
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
    for value in block.iter_mut() {
        *value = 0.0;
    }

    let mean_rgb = tensor.mean_rgb();
    let (mean_h_norm, mean_s, mean_l) = rgb_to_hsl(mean_rgb);
    let mean_h = canonical_hue((mean_h_norm * TAU).rem_euclid(TAU));
    if !block.is_empty() {
        block[0] = (mean_h / PI) - 1.0;
    }
    if block.len() > 1 {
        block[1] = mean_s.clamp(0.0, 1.0);
    }
    if block.len() > 2 {
        block[2] = mean_l.clamp(0.0, 1.0);
    }

    let samples = extract_hsl_samples(tensor);
    if samples.is_empty() {
        return;
    }

    let (_, _, layers, _) = tensor.shape();
    let mut per_layer = vec![HslStats::default(); layers.max(1)];

    for sample in &samples {
        if let Some(layer_stats) = per_layer.get_mut(sample.layer) {
            layer_stats.add(sample);
        }
    }

    let contexts = per_layer.len().min(3);
    for layer in 0..contexts {
        if let Some((h, s, l)) = per_layer[layer].means() {
            let offset = 3 + layer * 3;
            if block.len() > offset {
                block[offset] = (h / PI) - 1.0;
            }
            if block.len() > offset + 1 {
                block[offset + 1] = s;
            }
            if block.len() > offset + 2 {
                block[offset + 2] = l;
            }
        }
    }
}

fn extract_hsl_samples(tensor: &ChromaticTensor) -> Vec<HslSample> {
    let (rows, cols, layers, channels) = tensor.shape();
    debug_assert_eq!(channels, 3, "chromatic tensor must store RGB channels");

    let mut samples = Vec::with_capacity(rows * cols * layers);
    for layer in 0..layers {
        for row in 0..rows {
            for col in 0..cols {
                let rgb = [
                    tensor.colors[[row, col, layer, 0]],
                    tensor.colors[[row, col, layer, 1]],
                    tensor.colors[[row, col, layer, 2]],
                ];
                let (h_norm, saturation, luminance) = rgb_to_hsl(rgb);
                let hue = canonical_hue((h_norm * TAU).rem_euclid(TAU));
                samples.push(HslSample {
                    hue,
                    saturation,
                    luminance,
                    layer,
                });
            }
        }
    }

    samples
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

    #[test]
    fn hsl_feature_extraction_scans_full_grid() {
        let mut tensor = ChromaticTensor::new(3, 12, 12);
        for layer in 0..12 {
            for row in 0..3 {
                for col in 0..12 {
                    let base = (row + col + layer) as f32 / 27.0;
                    tensor.colors[[row, col, layer, 0]] = (base % 1.0).clamp(0.0, 1.0);
                    tensor.colors[[row, col, layer, 1]] = ((base * 0.5) % 1.0).clamp(0.0, 1.0);
                    tensor.colors[[row, col, layer, 2]] = ((base * 0.25) % 1.0).clamp(0.0, 1.0);
                }
            }
        }

        let samples = extract_hsl_samples(&tensor);
        assert_eq!(samples.len(), 3 * 12 * 12);

        let mut layer_counts = [0usize; 12];
        for sample in samples {
            layer_counts[sample.layer] += 1;
        }

        for count in layer_counts {
            assert_eq!(count, 3 * 12);
        }
    }

    #[test]
    fn encode_hsl_block_preserves_contextual_statistics() {
        let mut tensor = ChromaticTensor::new(3, 12, 12);
        let colors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        for layer in 0..12 {
            let color = colors.get(layer).copied().unwrap_or([0.5, 0.5, 0.5]);
            for row in 0..3 {
                for col in 0..12 {
                    for channel in 0..3 {
                        tensor.colors[[row, col, layer, channel]] = color[channel];
                    }
                }
            }
        }

        let mut block = [0.0f32; HSL_DIM];
        encode_hsl_block(&mut block, &tensor);

        assert!((block[0] + 1.0).abs() < 1e-6);
        assert!(block[1].abs() < 1e-6);
        assert!((block[2] - (11.0 / 24.0)).abs() < 1e-6);

        let red_offset = 3;
        assert!((block[red_offset] + 1.0).abs() < 1e-6);
        assert!((block[red_offset + 1] - 1.0).abs() < 1e-6);
        assert!((block[red_offset + 2] - 0.5).abs() < 1e-6);

        let green_offset = 6;
        assert!((block[green_offset] + 1.0 / 3.0).abs() < 1e-6);
        assert!((block[green_offset + 1] - 1.0).abs() < 1e-6);
        assert!((block[green_offset + 2] - 0.5).abs() < 1e-6);

        let blue_offset = 9;
        assert!((block[blue_offset] - 1.0 / 3.0).abs() < 1e-6);
        assert!((block[blue_offset + 1] - 1.0).abs() < 1e-6);
        assert!((block[blue_offset + 2] - 0.5).abs() < 1e-6);
    }
}
