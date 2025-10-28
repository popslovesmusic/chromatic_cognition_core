//! Engine configuration management via TOML files.
//!
//! This module provides configuration parsing from TOML format with sensible defaults.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use serde::Serialize;
use toml::Value;

use crate::meta::predict::Feature;

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
}
