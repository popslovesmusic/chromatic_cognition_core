//! Checkpoint trait and error handling for deterministic state persistence.
//!
//! This module provides a reusable [`Checkpointable`] trait that enforces a
//! deterministic, versioned serialization contract for engine components.
//! Implementations are responsible for storing a version header alongside the
//! serialized payload so that incompatible files are rejected during load.

use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use bincode::Options;

use crate::dream::error::DreamError;

/// Errors that can occur while saving or loading checkpoints.
#[derive(Debug)]
pub enum CheckpointError {
    /// Underlying I/O failure while reading or writing checkpoint files.
    Io(std::io::Error),
    /// Serialization or deserialization error from the binary codec.
    Serialization(bincode::Error),
    /// The checkpoint file was well formed but produced an incompatible schema version.
    VersionMismatch { expected: u32, found: u32 },
    /// The checkpoint file did not match the expected structure.
    InvalidFormat(String),
    /// Dream-specific error emitted while reconstructing runtime indexes.
    Dream(DreamError),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckpointError::Io(err) => write!(f, "I/O error while accessing checkpoint: {err}"),
            CheckpointError::Serialization(err) => {
                write!(f, "Failed to (de)serialize checkpoint payload: {err}")
            }
            CheckpointError::VersionMismatch { expected, found } => write!(
                f,
                "Checkpoint version mismatch: expected {expected}, found {found}",
            ),
            CheckpointError::InvalidFormat(msg) => {
                write!(f, "Checkpoint file has invalid structure: {msg}")
            }
            CheckpointError::Dream(err) => write!(f, "Failed to rebuild runtime indexes: {err}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<std::io::Error> for CheckpointError {
    fn from(err: std::io::Error) -> Self {
        CheckpointError::Io(err)
    }
}

impl From<bincode::Error> for CheckpointError {
    fn from(err: bincode::Error) -> Self {
        CheckpointError::Serialization(err)
    }
}

impl From<DreamError> for CheckpointError {
    fn from(err: DreamError) -> Self {
        CheckpointError::Dream(err)
    }
}

/// Deterministic binary codec options shared by all checkpoint implementations.
fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .with_little_endian()
}

/// Components that support deterministic persistence implement this trait.
pub trait Checkpointable: Sized {
    /// Save the current state to `path` using the deterministic codec.
    fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<(), CheckpointError>;

    /// Load a state from `path`, replacing any existing instance.
    fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<Self, CheckpointError>;

    /// Utility for writing a serializable snapshot with the shared codec.
    fn write_snapshot<P, T>(snapshot: &T, path: P) -> Result<(), CheckpointError>
    where
        P: AsRef<Path>,
        T: serde::Serialize,
    {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        codec().serialize_into(&mut writer, snapshot)?;
        writer.flush()?;
        Ok(())
    }

    /// Utility for reading a serializable snapshot with the shared codec.
    fn read_snapshot<P, T>(path: P) -> Result<T, CheckpointError>
    where
        P: AsRef<Path>,
        T: serde::de::DeserializeOwned,
    {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Ok(codec().deserialize_from(&mut reader)?)
    }
}
