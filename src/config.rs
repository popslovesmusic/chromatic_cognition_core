//! Engine configuration management via TOML files.
//!
//! This module provides configuration parsing from TOML format with sensible defaults.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use toml::Value;

use crate::meta::dissonance::DissonanceWeights;
use crate::meta::ethics::EthicsBounds;
use crate::meta::predict::Feature;
use crate::meta::reflection::ReflectionAction;

/// Engine configuration loaded from TOML file.
///
/// # Examples
///
/// ```
/// use chromatic_cognition_core::EngineConfig;
///
/// // Load from file
/// let config = EngineConfig::load_from_file("config/engine.toml")
///     .unwrap_or_else(|_| EngineConfig::default());
///
/// println!("Tensor dimensions: {}x{}x{}", config.rows, config.cols, config.layers);
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct EngineConfig {
    /// Number of rows in the tensor grid
    pub rows: usize,
    /// Number of columns in the tensor grid
    pub cols: usize,
    /// Number of depth layers in the tensor
    pub layers: usize,
    /// Random seed for deterministic initialization
    pub seed: u64,
    /// Target device ("cpu" or future "cuda"/"metal")
    pub device: String,
}

impl EngineConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(&path)?;
        Self::from_str(&contents)
    }

    pub fn from_str(toml_str: &str) -> Result<Self, ConfigError> {
        let mut section = String::new();
        let mut values: HashMap<String, String> = HashMap::new();

        for line in toml_str.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                section = trimmed.trim_matches(&['[', ']'][..]).to_string();
                continue;
            }

            let (key, value) = trimmed
                .split_once('=')
                .ok_or_else(|| ConfigError::Parse(format!("Invalid line: {}", trimmed)))?;
            let key = key.trim().to_string();
            let value = value.trim().trim_matches('"').to_string();
            values.insert(format!("{}::{}", section, key), value);
        }

        let rows = values
            .remove("engine::rows")
            .map(|v| v.parse())
            .transpose()
            .map_err(|_| ConfigError::Parse("rows must be an integer".into()))?
            .unwrap_or(64);
        let cols = values
            .remove("engine::cols")
            .map(|v| v.parse())
            .transpose()
            .map_err(|_| ConfigError::Parse("cols must be an integer".into()))?
            .unwrap_or(64);
        let layers = values
            .remove("engine::layers")
            .map(|v| v.parse())
            .transpose()
            .map_err(|_| ConfigError::Parse("layers must be an integer".into()))?
            .unwrap_or(8);
        let seed = values
            .remove("engine::seed")
            .map(|v| v.parse())
            .transpose()
            .map_err(|_| ConfigError::Parse("seed must be an integer".into()))?
            .unwrap_or(42);
        let device = values
            .remove("engine::device")
            .unwrap_or_else(|| "cpu".to_string());

        Ok(Self {
            rows,
            cols,
            layers,
            seed,
            device,
        })
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            rows: 64,
            cols: 64,
            layers: 8,
            seed: 42,
            device: "cpu".to_string(),
        }
    }
}

/// Bridge configuration describing the chromatic ↔ sonic mapping parameters.
#[derive(Debug, Clone, Serialize)]
pub struct BridgeConfig {
    /// Base hue → frequency mapping parameters.
    pub base: BridgeBaseConfig,
    /// Spectral accumulation and reduction settings.
    pub spectral: BridgeSpectralConfig,
    /// Reversibility guard-rail thresholds.
    pub reversibility: BridgeReversibilityConfig,
}

impl BridgeConfig {
    /// Load bridge configuration from a TOML file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(path)?;
        Self::from_str(&contents)
    }

    /// Parse bridge configuration from a TOML string.
    pub fn from_str(toml_str: &str) -> Result<Self, ConfigError> {
        let raw: RawBridgeConfig =
            toml::from_str(toml_str).map_err(|err| ConfigError::Parse(err.to_string()))?;

        let base = BridgeBaseConfig::try_from(&raw.bridge)?;
        let spectral = BridgeSpectralConfig::try_from(&raw.bridge.spectral)?;
        let reversibility = BridgeReversibilityConfig::try_from(&raw.bridge.reversibility)?;

        Ok(Self {
            base,
            spectral,
            reversibility,
        })
    }
}

/// Base hue → frequency configuration parameters.
#[derive(Debug, Clone, Serialize)]
pub struct BridgeBaseConfig {
    pub f_min: f32,
    pub octaves: f32,
    pub gamma: f32,
    pub sample_rate: u32,
}

impl BridgeBaseConfig {
    fn try_from(raw: &RawBridgeBase) -> Result<Self, ConfigError> {
        if !raw.f_min.is_finite() || raw.f_min <= 0.0 {
            return Err(ConfigError::Parse("bridge.f_min must be positive".into()));
        }
        if !raw.octaves.is_finite() || raw.octaves <= 0.0 {
            return Err(ConfigError::Parse("bridge.octaves must be positive".into()));
        }
        if !raw.gamma.is_finite() || raw.gamma <= 0.0 {
            return Err(ConfigError::Parse("bridge.gamma must be positive".into()));
        }
        if raw.sample_rate == 0 {
            return Err(ConfigError::Parse(
                "bridge.sample_rate must be non-zero".into(),
            ));
        }

        Ok(Self {
            f_min: raw.f_min,
            octaves: raw.octaves,
            gamma: raw.gamma,
            sample_rate: raw.sample_rate,
        })
    }
}

/// Spectral accumulation configuration parameters.
#[derive(Debug, Clone, Serialize)]
pub struct BridgeSpectralConfig {
    pub fft_size: usize,
    pub accum_format: String,
    pub reduction_mode: String,
    pub categorical_count: usize,
}

impl BridgeSpectralConfig {
    fn try_from(raw: &RawBridgeSpectral) -> Result<Self, ConfigError> {
        if raw.fft_size < 2 {
            return Err(ConfigError::Parse(
                "bridge.spectral.fft_size must be ≥ 2".into(),
            ));
        }
        if raw.categorical_count == 0 {
            return Err(ConfigError::Parse(
                "bridge.spectral.categorical_count must be ≥ 1".into(),
            ));
        }

        Ok(Self {
            fft_size: raw.fft_size,
            accum_format: raw.accum_format.clone(),
            reduction_mode: raw.reduction_mode.clone(),
            categorical_count: raw.categorical_count,
        })
    }
}

/// Reversibility guard-rail configuration.
#[derive(Debug, Clone, Serialize)]
pub struct BridgeReversibilityConfig {
    pub delta_e_tolerance: f32,
}

impl BridgeReversibilityConfig {
    fn try_from(raw: &RawBridgeReversibility) -> Result<Self, ConfigError> {
        if !raw.delta_e_tolerance.is_finite() || raw.delta_e_tolerance < 0.0 {
            return Err(ConfigError::Parse(
                "bridge.reversibility.delta_e_tolerance must be ≥ 0".into(),
            ));
        }

        Ok(Self {
            delta_e_tolerance: raw.delta_e_tolerance,
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawBridgeConfig {
    bridge: RawBridgeBase,
}

#[derive(Debug, Deserialize)]
struct RawBridgeBase {
    #[serde(default = "default_f_min")]
    f_min: f32,
    #[serde(default = "default_octaves")]
    octaves: f32,
    #[serde(default = "default_gamma")]
    gamma: f32,
    #[serde(default = "default_sample_rate")]
    sample_rate: u32,
    #[serde(default)]
    spectral: RawBridgeSpectral,
    #[serde(default)]
    reversibility: RawBridgeReversibility,
}

#[derive(Debug, Deserialize)]
struct RawBridgeSpectral {
    #[serde(default = "default_fft_size")]
    fft_size: usize,
    #[serde(default = "default_accum_format")]
    accum_format: String,
    #[serde(default = "default_reduction_mode")]
    reduction_mode: String,
    #[serde(default = "default_categorical_count")]
    categorical_count: usize,
}

impl Default for RawBridgeSpectral {
    fn default() -> Self {
        Self {
            fft_size: default_fft_size(),
            accum_format: default_accum_format(),
            reduction_mode: default_reduction_mode(),
            categorical_count: default_categorical_count(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawBridgeReversibility {
    #[serde(default = "default_delta_e_tolerance")]
    delta_e_tolerance: f32,
}

impl Default for RawBridgeReversibility {
    fn default() -> Self {
        Self {
            delta_e_tolerance: default_delta_e_tolerance(),
        }
    }
}

fn default_f_min() -> f32 {
    110.0
}

fn default_octaves() -> f32 {
    6.0
}

fn default_gamma() -> f32 {
    1.0
}

fn default_sample_rate() -> u32 {
    44_100
}

fn default_fft_size() -> usize {
    1024
}

fn default_accum_format() -> String {
    "Q16.48".to_string()
}

fn default_reduction_mode() -> String {
    "pairwise_neumaier".to_string()
}

fn default_categorical_count() -> usize {
    12
}

fn default_delta_e_tolerance() -> f32 {
    1.0e-3
}

#[derive(Debug, Clone, Serialize)]
pub struct Phase5AConfig {
    pub predict_horizon: usize,
    pub features: Vec<Feature>,
}

impl Phase5AConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(&path)?;
        Self::from_str(&contents)
    }

    pub fn from_str(toml_str: &str) -> Result<Self, ConfigError> {
        let value: Value =
            toml::from_str(toml_str).map_err(|err| ConfigError::Parse(err.to_string()))?;
        let table = value
            .get("p5a")
            .and_then(|v| v.as_table())
            .cloned()
            .unwrap_or_default();

        let horizon = table
            .get("predict_horizon")
            .and_then(|v| v.as_integer())
            .map(|v| v.max(1) as usize)
            .unwrap_or(2)
            .clamp(1, 16);

        let features = table
            .get("features")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str())
                    .filter_map(|name| Feature::from_str(name).ok())
                    .collect::<Vec<_>>()
            })
            .filter(|features| !features.is_empty())
            .unwrap_or_else(Self::default_features);

        Ok(Self {
            predict_horizon: horizon,
            features,
        })
    }

    fn default_features() -> Vec<Feature> {
        vec![Feature::Coherence, Feature::Entropy, Feature::GradEnergy]
    }
}

impl Default for Phase5AConfig {
    fn default() -> Self {
        Self {
            predict_horizon: 2,
            features: Self::default_features(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Phase5BConfig {
    pub dissonance_threshold: f32,
    pub weights: DissonanceWeights,
    pub actions: Vec<ReflectionAction>,
}

impl Phase5BConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(&path)?;
        Self::from_str(&contents)
    }

    pub fn from_str(toml_str: &str) -> Result<Self, ConfigError> {
        let value: Value =
            toml::from_str(toml_str).map_err(|err| ConfigError::Parse(err.to_string()))?;
        let table = value
            .get("p5b")
            .and_then(|v| v.as_table())
            .cloned()
            .unwrap_or_default();

        let threshold = table
            .get("dissonance_threshold")
            .and_then(|v| v.as_float())
            .map(|v| (v as f32).clamp(0.0, 1.0))
            .unwrap_or(0.25);

        let weights = table
            .get("weights")
            .and_then(|value| value.as_table())
            .map(|weights| DissonanceWeights {
                coherence: weights
                    .get("coherence")
                    .and_then(|v| v.as_float())
                    .map(|v| v as f32)
                    .unwrap_or(0.5),
                entropy: weights
                    .get("entropy")
                    .and_then(|v| v.as_float())
                    .map(|v| v as f32)
                    .unwrap_or(0.3),
                energy: weights
                    .get("energy")
                    .and_then(|v| v.as_float())
                    .map(|v| v as f32)
                    .unwrap_or(0.2),
            })
            .unwrap_or_else(Self::default_weights);

        let actions = table
            .get("actions")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str())
                    .filter_map(|name| name.parse::<ReflectionAction>().ok())
                    .collect::<Vec<_>>()
            })
            .filter(|actions| !actions.is_empty())
            .unwrap_or_else(Self::default_actions);

        Ok(Self {
            dissonance_threshold: threshold,
            weights: weights.normalised(),
            actions,
        })
    }

    fn default_weights() -> DissonanceWeights {
        DissonanceWeights {
            coherence: 0.5,
            entropy: 0.3,
            energy: 0.2,
        }
    }

    fn default_actions() -> Vec<ReflectionAction> {
        vec![
            ReflectionAction::SeedFrom,
            ReflectionAction::DampLr,
            ReflectionAction::CoolTint,
            ReflectionAction::PauseAug,
        ]
    }
}

impl Default for Phase5BConfig {
    fn default() -> Self {
        Self {
            dissonance_threshold: 0.25,
            weights: Self::default_weights().normalised(),
            actions: Self::default_actions(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Phase5CConfig {
    pub bounds: EthicsBounds,
    pub log_every: usize,
}

impl Phase5CConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(&path)?;
        Self::from_str(&contents)
    }

    pub fn from_str(toml_str: &str) -> Result<Self, ConfigError> {
        let value: Value =
            toml::from_str(toml_str).map_err(|err| ConfigError::Parse(err.to_string()))?;
        let table = value
            .get("p5c")
            .and_then(|v| v.as_table())
            .cloned()
            .unwrap_or_default();

        let lr_damp_max = table
            .get("lr_damp_max")
            .and_then(|v| v.as_float())
            .map(|v| (v as f32).clamp(0.0, 1.0))
            .unwrap_or(0.5);

        let cool_tint_max = table
            .get("cool_tint_max")
            .and_then(|v| v.as_float())
            .map(|v| (v as f32).clamp(0.0, 1.0))
            .unwrap_or(0.2);

        let pause_aug_max_steps = table
            .get("pause_aug_max_steps")
            .and_then(|v| v.as_integer())
            .map(|v| v.max(0) as usize)
            .unwrap_or(200);

        let ethics_hue_jump_deg = table
            .get("ethics_hue_jump_deg")
            .map(|value| {
                if let Some(float) = value.as_float() {
                    float as f32
                } else if let Some(int) = value.as_integer() {
                    int as f32
                } else {
                    90.0
                }
            })
            .unwrap_or(90.0)
            .max(0.0);

        let log_every = table
            .get("log_every")
            .and_then(|v| v.as_integer())
            .map(|v| v.max(1) as usize)
            .unwrap_or(1);

        Ok(Self {
            bounds: EthicsBounds {
                lr_damp_max,
                cool_tint_max,
                pause_aug_max_steps,
                ethics_hue_jump_deg,
            },
            log_every,
        })
    }
}

impl Default for Phase5CConfig {
    fn default() -> Self {
        Self {
            bounds: EthicsBounds {
                lr_damp_max: 0.5,
                cool_tint_max: 0.2,
                pause_aug_max_steps: 200,
                ethics_hue_jump_deg: 90.0,
            },
            log_every: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Phase6CConfig {
    pub cycle_interval: usize,
    pub lr_adjust_max: f32,
    pub dream_pool_expand_max: usize,
    pub trend_anomaly_cooldown: usize,
}

impl Phase6CConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(&path)?;
        Self::from_str(&contents)
    }

    pub fn from_str(toml_str: &str) -> Result<Self, ConfigError> {
        let value: Value =
            toml::from_str(toml_str).map_err(|err| ConfigError::Parse(err.to_string()))?;
        let table = value
            .get("p6c")
            .and_then(|v| v.as_table())
            .cloned()
            .unwrap_or_default();

        let cycle_interval = table
            .get("cycle_interval")
            .and_then(|v| v.as_integer())
            .map(|v| v.max(1) as usize)
            .unwrap_or(10);

        let lr_adjust_max = table
            .get("lr_adjust_max")
            .and_then(|v| v.as_float())
            .map(|v| (v as f32).max(0.0))
            .unwrap_or(0.2);

        let dream_pool_expand_max = table
            .get("dream_pool_expand_max")
            .and_then(|v| v.as_integer())
            .map(|v| v.max(0) as usize)
            .unwrap_or(50);

        let trend_anomaly_cooldown = table
            .get("trend_anomaly_cooldown")
            .and_then(|v| v.as_integer())
            .map(|v| v.max(0) as usize)
            .unwrap_or(5);

        Ok(Self {
            cycle_interval,
            lr_adjust_max,
            dream_pool_expand_max,
            trend_anomaly_cooldown,
        })
    }
}

impl Default for Phase6CConfig {
    fn default() -> Self {
        Self {
            cycle_interval: 10,
            lr_adjust_max: 0.2,
            dream_pool_expand_max: 50,
            trend_anomaly_cooldown: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Phase6DRiskWeight {
    pub loss: f32,
    pub entropy: f32,
    pub coherence: f32,
    pub oscillation: f32,
}

impl Phase6DRiskWeight {
    pub fn normalized(&self) -> Self {
        let sum = self.loss + self.entropy + self.coherence + self.oscillation;
        if sum <= f32::EPSILON {
            return Self::default();
        }

        Self {
            loss: (self.loss / sum).clamp(0.0, 1.0),
            entropy: (self.entropy / sum).clamp(0.0, 1.0),
            coherence: (self.coherence / sum).clamp(0.0, 1.0),
            oscillation: (self.oscillation / sum).clamp(0.0, 1.0),
        }
    }
}

impl Default for Phase6DRiskWeight {
    fn default() -> Self {
        Self {
            loss: 0.4,
            entropy: 0.3,
            coherence: 0.2,
            oscillation: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Phase6DConfig {
    pub loss_slope_limit: f32,
    pub entropy_drift_limit: f32,
    pub coherence_decay_limit: f32,
    pub oscillation_index_limit: f32,
    pub risk_weight: Phase6DRiskWeight,
    pub action_delay: usize,
}

impl Phase6DConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(&path)?;
        Self::from_str(&contents)
    }

    pub fn from_str(toml_str: &str) -> Result<Self, ConfigError> {
        let value: Value =
            toml::from_str(toml_str).map_err(|err| ConfigError::Parse(err.to_string()))?;
        let table = value
            .get("p6d")
            .and_then(|v| v.as_table())
            .cloned()
            .unwrap_or_default();

        let loss_slope_limit = table
            .get("loss_slope_limit")
            .and_then(|v| v.as_float())
            .map(|v| (v as f32).max(1e-6))
            .unwrap_or(0.03);

        let entropy_drift_limit = table
            .get("entropy_drift_limit")
            .and_then(|v| v.as_float())
            .map(|v| (v as f32).max(1e-6))
            .unwrap_or(0.03);

        let coherence_decay_limit = table
            .get("coherence_decay_limit")
            .and_then(|v| v.as_float())
            .map(|v| (v as f32).max(1e-6))
            .unwrap_or(0.02);

        let oscillation_index_limit = table
            .get("oscillation_index_limit")
            .and_then(|v| v.as_float())
            .map(|v| (v as f32).max(1e-6))
            .unwrap_or(0.1);

        let risk_weight_table = table
            .get("risk_weight")
            .and_then(|v| v.as_table())
            .cloned()
            .unwrap_or_default();

        let risk_weight = Phase6DRiskWeight {
            loss: risk_weight_table
                .get("loss")
                .and_then(|v| v.as_float())
                .map(|v| v as f32)
                .unwrap_or(0.4),
            entropy: risk_weight_table
                .get("entropy")
                .and_then(|v| v.as_float())
                .map(|v| v as f32)
                .unwrap_or(0.3),
            coherence: risk_weight_table
                .get("coherence")
                .and_then(|v| v.as_float())
                .map(|v| v as f32)
                .unwrap_or(0.2),
            oscillation: risk_weight_table
                .get("oscillation")
                .and_then(|v| v.as_float())
                .map(|v| v as f32)
                .unwrap_or(0.1),
        };

        let action_delay = table
            .get("action_delay")
            .and_then(|v| v.as_integer())
            .map(|v| v.max(0) as usize)
            .unwrap_or(2);

        Ok(Self {
            loss_slope_limit,
            entropy_drift_limit,
            coherence_decay_limit,
            oscillation_index_limit,
            risk_weight,
            action_delay,
        })
    }
}

impl Default for Phase6DConfig {
    fn default() -> Self {
        Self {
            loss_slope_limit: 0.03,
            entropy_drift_limit: 0.03,
            coherence_decay_limit: 0.02,
            oscillation_index_limit: 0.1,
            risk_weight: Phase6DRiskWeight::default(),
            action_delay: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Phase6BConfig {
    pub trend_window: usize,
    pub trend_drift_limit: f32,
    pub oscillation_limit: f32,
}

impl Phase6BConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(&path)?;
        Self::from_str(&contents)
    }

    pub fn from_str(toml_str: &str) -> Result<Self, ConfigError> {
        let value: Value =
            toml::from_str(toml_str).map_err(|err| ConfigError::Parse(err.to_string()))?;
        let table = value
            .get("p6b")
            .and_then(|v| v.as_table())
            .cloned()
            .unwrap_or_default();

        let trend_window = table
            .get("trend_window")
            .and_then(|v| v.as_integer())
            .map(|v| v.max(2) as usize)
            .unwrap_or(20)
            .min(512);

        let trend_drift_limit = table
            .get("trend_drift_limit")
            .and_then(|v| v.as_float())
            .map(|v| (v as f32).max(0.0))
            .unwrap_or(0.03);

        let oscillation_limit = table
            .get("oscillation_limit")
            .and_then(|v| v.as_float())
            .map(|v| (v as f32).clamp(0.0, 1.0))
            .unwrap_or(0.15);

        Ok(Self {
            trend_window,
            trend_drift_limit,
            oscillation_limit,
        })
    }
}

impl Default for Phase6BConfig {
    fn default() -> Self {
        Self {
            trend_window: 20,
            trend_drift_limit: 0.03,
            oscillation_limit: 0.15,
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Parse(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(err) => write!(f, "IO error: {}", err),
            ConfigError::Parse(err) => write!(f, "Parse error: {}", err),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(value: std::io::Error) -> Self {
        ConfigError::Io(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase5a_config_defaults_when_section_missing() {
        let toml = "[engine]\nrows = 8";
        let config = Phase5AConfig::from_str(toml).unwrap();
        assert_eq!(config.predict_horizon, 2);
        assert_eq!(config.features, Phase5AConfig::default().features);
    }

    #[test]
    fn phase5a_config_parses_custom_values() {
        let toml = "[p5a]\npredict_horizon = 4\nfeatures = [\"coherence\", \"grad_energy\"]";
        let config = Phase5AConfig::from_str(toml).unwrap();
        assert_eq!(config.predict_horizon, 4);
        assert_eq!(
            config.features,
            vec![Feature::Coherence, Feature::GradEnergy]
        );
    }

    #[test]
    fn phase5c_config_defaults_when_missing() {
        let toml = "[engine]\nrows = 8";
        let config = Phase5CConfig::from_str(toml).unwrap();
        assert_eq!(config.bounds.lr_damp_max, 0.5);
        assert_eq!(config.log_every, 1);
    }

    #[test]
    fn phase5c_config_parses_custom_values() {
        let toml = "[p5c]\nlr_damp_max = 0.4\ncool_tint_max = 0.1\npause_aug_max_steps = 50\nethics_hue_jump_deg = 60\nlog_every = 3";
        let config = Phase5CConfig::from_str(toml).unwrap();
        assert_eq!(config.bounds.lr_damp_max, 0.4);
        assert_eq!(config.bounds.cool_tint_max, 0.1);
        assert_eq!(config.bounds.pause_aug_max_steps, 50);
        assert_eq!(config.bounds.ethics_hue_jump_deg, 60.0);
        assert_eq!(config.log_every, 3);
    }

    #[test]
    fn phase6c_config_defaults_when_missing() {
        let toml = "[engine]\nrows = 8";
        let config = Phase6CConfig::from_str(toml).unwrap();
        assert_eq!(config.cycle_interval, 10);
        assert!((config.lr_adjust_max - 0.2).abs() < f32::EPSILON);
        assert_eq!(config.dream_pool_expand_max, 50);
        assert_eq!(config.trend_anomaly_cooldown, 5);
    }

    #[test]
    fn phase6c_config_parses_custom_values() {
        let toml = "[p6c]\ncycle_interval = 6\nlr_adjust_max = 0.15\ndream_pool_expand_max = 12\ntrend_anomaly_cooldown = 9";
        let config = Phase6CConfig::from_str(toml).unwrap();
        assert_eq!(config.cycle_interval, 6);
        assert!((config.lr_adjust_max - 0.15).abs() < f32::EPSILON);
        assert_eq!(config.dream_pool_expand_max, 12);
        assert_eq!(config.trend_anomaly_cooldown, 9);
    }

    #[test]
    fn phase6b_config_defaults_when_missing() {
        let toml = "[engine]\nrows = 8";
        let config = Phase6BConfig::from_str(toml).unwrap();
        assert_eq!(config.trend_window, 20);
        assert!((config.trend_drift_limit - 0.03).abs() < f32::EPSILON);
        assert!((config.oscillation_limit - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn phase6b_config_parses_custom_values() {
        let toml = "[p6b]\ntrend_window = 32\ntrend_drift_limit = 0.05\noscillation_limit = 0.2";
        let config = Phase6BConfig::from_str(toml).unwrap();
        assert_eq!(config.trend_window, 32);
        assert!((config.trend_drift_limit - 0.05).abs() < f32::EPSILON);
        assert!((config.oscillation_limit - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn phase6d_config_defaults_when_missing() {
        let toml = "[engine]\nrows = 8";
        let config = Phase6DConfig::from_str(toml).unwrap();
        assert!((config.loss_slope_limit - 0.03).abs() < f32::EPSILON);
        assert_eq!(config.action_delay, 2);
        let weights = config.risk_weight.normalized();
        let sum = weights.loss + weights.entropy + weights.coherence + weights.oscillation;
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn phase6d_config_parses_custom_values() {
        let toml = "[p6d]\nloss_slope_limit = 0.05\nentropy_drift_limit = 0.04\ncoherence_decay_limit = 0.03\noscillation_index_limit = 0.2\naction_delay = 4\n[p6d.risk_weight]\nloss = 0.3\nentropy = 0.4\ncoherence = 0.2\noscillation = 0.1";
        let config = Phase6DConfig::from_str(toml).unwrap();
        assert!((config.loss_slope_limit - 0.05).abs() < f32::EPSILON);
        assert!((config.entropy_drift_limit - 0.04).abs() < f32::EPSILON);
        assert!((config.coherence_decay_limit - 0.03).abs() < f32::EPSILON);
        assert!((config.oscillation_index_limit - 0.2).abs() < f32::EPSILON);
        assert_eq!(config.action_delay, 4);
        assert!((config.risk_weight.loss - 0.3).abs() < f32::EPSILON);
        assert!((config.risk_weight.entropy - 0.4).abs() < f32::EPSILON);
        assert!((config.risk_weight.coherence - 0.2).abs() < f32::EPSILON);
        assert!((config.risk_weight.oscillation - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn bridge_config_parses_full_spec() {
        let toml = r#"
[bridge]
f_min = 110.0
octaves = 7
gamma = 1.0
sample_rate = 44100

[bridge.spectral]
fft_size = 4096
accum_format = "Q16.48"
reduction_mode = "pairwise_neumaier"
categorical_count = 12

[bridge.reversibility]
delta_e_tolerance = 0.001
"#;
        let config = BridgeConfig::from_str(toml).unwrap();
        assert!((config.base.f_min - 110.0).abs() < f32::EPSILON);
        assert_eq!(config.base.sample_rate, 44_100);
        assert_eq!(config.spectral.fft_size, 4096);
        assert_eq!(config.spectral.accum_format, "Q16.48");
        assert_eq!(config.reversibility.delta_e_tolerance, 0.001);
    }

    #[test]
    fn bridge_config_rejects_invalid_fft_size() {
        let toml = r#"
[bridge]
[bridge.spectral]
fft_size = 1
"#;
        let result = BridgeConfig::from_str(toml);
        assert!(result.is_err());
    }
}
