//! Meta-layer logging utilities for Phase 5 self-regulation.
//!
//! The logger records every ethics-reviewed intervention in a JSONL journal
//! while keeping an in-memory buffer for deterministic testing.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use super::reflection::ReflectionAction;

/// Serialisable snapshot of the training controls after an intervention.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ControlStateSnapshot {
    pub learning_rate: f32,
    pub tint_cool_strength: f32,
    pub hue_shift_deg: f32,
    pub augmentation_pause_steps: usize,
    pub reseed_from_step: Option<usize>,
}

/// Status of a logged meta action.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum MetaActionStatus {
    Applied,
    Clipped,
    Rejected,
    Skipped,
    Rollback,
}

/// Single journal entry emitted by the meta logger.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MetaLogEntry {
    pub sequence: usize,
    pub step: usize,
    pub action: Option<ReflectionAction>,
    pub status: MetaActionStatus,
    pub details: String,
    pub timestamp_ms: u128,
    pub state: ControlStateSnapshot,
}

fn ensure_log_dir() -> io::Result<()> {
    fs::create_dir_all("logs")
}

fn append_json_line<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, value)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
    file.write_all(b"\n")
}

/// Deterministic meta logger with configurable sampling interval.
#[derive(Debug, Clone, Serialize)]
pub struct MetaLogger {
    log_every: usize,
    sequence: usize,
    entries: Vec<MetaLogEntry>,
}

impl MetaLogger {
    pub fn new(log_every: usize) -> Self {
        Self {
            log_every: log_every.max(1),
            sequence: 0,
            entries: Vec::new(),
        }
    }

    pub fn entries(&self) -> &[MetaLogEntry] {
        &self.entries
    }

    pub fn record(&mut self, mut entry: MetaLogEntry) -> io::Result<()> {
        self.sequence += 1;
        entry.sequence = self.sequence;
        self.entries.push(entry.clone());

        if self.sequence % self.log_every == 0 {
            ensure_log_dir()?;
            append_json_line("logs/meta.jsonl", &entry)?;
        }

        Ok(())
    }

    pub fn timestamp_now() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logger_records_sequence_in_order() {
        let mut logger = MetaLogger::new(2);
        for idx in 0..5 {
            let entry = MetaLogEntry {
                sequence: 0,
                step: idx,
                action: None,
                status: MetaActionStatus::Skipped,
                details: format!("cycle {}", idx),
                timestamp_ms: 0,
                state: ControlStateSnapshot {
                    learning_rate: 0.1,
                    tint_cool_strength: 0.0,
                    hue_shift_deg: 0.0,
                    augmentation_pause_steps: 0,
                    reseed_from_step: None,
                },
            };
            logger.record(entry).unwrap();
        }

        let sequences: Vec<usize> = logger.entries().iter().map(|e| e.sequence).collect();
        assert_eq!(sequences, vec![1, 2, 3, 4, 5]);
        assert_eq!(logger.entries().len(), 5);
    }
}
