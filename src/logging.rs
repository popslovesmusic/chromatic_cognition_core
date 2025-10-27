use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::tensor::TensorStatistics;

fn log_dir() -> io::Result<()> {
    fs::create_dir_all("logs")
}

fn append_json_line<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, value)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
    file.write_all(b"\n")
}

#[derive(Debug, Serialize)]
pub struct OperationLogEntry {
    pub operation: String,
    pub timestamp_ms: u128,
    pub mean_rgb: [f32; 3],
    pub variance: f32,
    pub certainty_mean: f32,
}

pub fn log_operation(operation: &str, stats: &TensorStatistics) -> io::Result<()> {
    log_dir()?;
    let entry = OperationLogEntry {
        operation: operation.to_string(),
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        mean_rgb: stats.mean_rgb,
        variance: stats.variance,
        certainty_mean: stats.mean_certainty,
    };
    append_json_line("logs/operations.jsonl", &entry)
}

#[derive(Debug, Serialize)]
pub struct TrainingLogEntry {
    pub iteration: usize,
    pub loss: f32,
    pub mean_rgb: [f32; 3],
    pub variance: f32,
    pub timestamp_ms: u128,
}

pub fn log_training_iteration(
    iteration: usize,
    metrics: &crate::training::TrainingMetrics,
) -> io::Result<()> {
    log_dir()?;
    let entry = TrainingLogEntry {
        iteration,
        loss: metrics.loss,
        mean_rgb: metrics.mean_rgb,
        variance: metrics.variance,
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    };
    append_json_line("logs/run.jsonl", &entry)
}
