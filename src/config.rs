use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct EngineConfig {
    pub rows: usize,
    pub cols: usize,
    pub layers: usize,
    pub seed: u64,
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
