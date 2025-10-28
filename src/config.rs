//! Engine configuration management via TOML files.
//!
//! This module provides configuration parsing from TOML format with sensible defaults.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use serde::Serialize;
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
}
