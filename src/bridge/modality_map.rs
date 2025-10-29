use std::f32::consts::PI;

use serde::Serialize;

use crate::{
    config::BridgeConfig,
    spectral::{canonical_hue, SpectralTensor},
    tensor::ChromaticTensor,
};

const DEFAULT_SEAM_EPSILON: f32 = PI * 0.05;

#[derive(Debug, Clone, Serialize)]
pub struct ModalityMapper {
    config: BridgeConfig,
}

impl ModalityMapper {
    pub fn new(config: BridgeConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &BridgeConfig {
        &self.config
    }

    pub fn encode_to_spectral(&self, chromatic: &ChromaticTensor) -> SpectralTensor {
        if let Err(err) =
            crate::logging::log_operation("modality_map_encode", &chromatic.statistics())
        {
            eprintln!("failed to log modality_map_encode: {err}");
        }

        SpectralTensor::from_chromatic_with_epsilon(
            chromatic,
            self.config.base.f_min,
            self.config.base.octaves,
            seam_epsilon(&self.config),
        )
    }

    pub fn decode_to_chromatic(&self, spectral: &SpectralTensor) -> ChromaticTensor {
        let chromatic = spectral.to_chromatic();
        if let Err(err) =
            crate::logging::log_operation("modality_map_decode", &chromatic.statistics())
        {
            eprintln!("failed to log modality_map_decode: {err}");
        }
        chromatic
    }

    pub fn map_hue_to_category(&self, hue_radians: f32) -> usize {
        let categories = self.config.spectral.categorical_count.max(1);
        let canonical = canonical_hue(hue_radians);
        let normalized = (canonical / std::f32::consts::TAU).clamp(0.0, 0.999_999);
        let scaled = normalized.mul_add(categories as f32, f32::EPSILON * 128.0);
        let mut index = scaled.floor() as usize;
        if index >= categories {
            index = categories - 1;
        }
        index
    }
}

fn seam_epsilon(config: &BridgeConfig) -> f32 {
    let categories = config.spectral.categorical_count.max(1) as f32;
    let suggested = (2.0 * PI) / (categories * 4.0);
    suggested.clamp(PI * 0.01, DEFAULT_SEAM_EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::BridgeConfig, tensor::ChromaticTensor};

    #[test]
    fn wraps_existing_spectral_mapping() {
        let config = BridgeConfig::from_str(
            r#"
            [bridge]
            f_min = 110.0
            octaves = 6.0
            gamma = 1.0

            [bridge.spectral]
            fft_size = 1024
            categorical_count = 12

            [bridge.reversibility]
            delta_e_tolerance = 1e-3
            "#,
        )
        .expect("valid bridge config");
        let mapper = ModalityMapper::new(config.clone());
        let chromatic = ChromaticTensor::from_seed(7, 4, 4, 2);

        let spectral_direct = SpectralTensor::from_chromatic_with_epsilon(
            &chromatic,
            config.base.f_min,
            config.base.octaves,
            seam_epsilon(&config),
        );
        let spectral_wrapped = mapper.encode_to_spectral(&chromatic);
        assert_eq!(spectral_direct.components, spectral_wrapped.components);
        assert_eq!(spectral_direct.certainty, spectral_wrapped.certainty);

        let decoded = mapper.decode_to_chromatic(&spectral_wrapped);
        let roundtrip = spectral_wrapped.to_chromatic();
        let (rows, cols, layers, _) = chromatic.shape();
        for row in 0..rows {
            for col in 0..cols {
                for layer in 0..layers {
                    let original = roundtrip.get_rgb(row, col, layer);
                    let mapped = decoded.get_rgb(row, col, layer);
                    for channel in 0..3 {
                        assert!((original[channel] - mapped[channel]).abs() < 1e-6);
                    }
                }
            }
        }
    }

    #[test]
    fn hue_to_category_respects_wraparound() {
        let config = BridgeConfig::from_str(
            r#"
            [bridge]
            f_min = 110.0
            octaves = 6.0
            gamma = 1.0

            [bridge.spectral]
            fft_size = 1024
            categorical_count = 12

            [bridge.reversibility]
            delta_e_tolerance = 1e-3
            "#,
        )
        .expect("valid bridge config");

        let mapper = ModalityMapper::new(config);
        assert_eq!(mapper.map_hue_to_category(0.0), 0);

        let wrapped = mapper.map_hue_to_category(std::f32::consts::TAU * 2.25);
        let canonical = mapper.map_hue_to_category(std::f32::consts::TAU * 0.25);
        assert_eq!(wrapped, canonical);

        let categories = mapper.config().spectral.categorical_count;
        let boundary = mapper.map_hue_to_category(std::f32::consts::TAU - 1e-6);
        assert_eq!(boundary, categories - 1);
    }
}
