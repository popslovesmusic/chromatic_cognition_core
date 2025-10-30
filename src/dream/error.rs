//! Error types for dream pool operations
//!
//! This module provides comprehensive error handling for all dream pool
//! operations, replacing panics with Result types for better library ergonomics.

use std::fmt;

/// Result type alias for dream pool operations
pub type DreamResult<T> = Result<T, DreamError>;

/// Comprehensive error type for dream pool operations
#[derive(Debug, Clone, PartialEq)]
pub enum DreamError {
    /// Dimension mismatch between query and index
    DimensionMismatch {
        expected: usize,
        got: usize,
        context: String,
    },

    /// Index has not been built yet
    IndexNotBuilt { operation: String },

    /// Pool has reached capacity
    CapacityExceeded { current: usize, max: usize },

    /// Invalid configuration parameter
    InvalidConfiguration {
        parameter: String,
        value: String,
        reason: String,
    },

    /// Empty pool or candidate set
    EmptyCollection { collection: String },

    /// Invalid parameter value
    InvalidParameter {
        parameter: String,
        value: String,
        constraint: String,
    },

    /// Index corruption detected
    IndexCorrupted { details: String },

    /// Memory budget exceeded
    MemoryExceeded { requested: usize, available: usize },

    /// Operation requires a feature that is not available
    FeatureUnavailable { feature: String, reason: String },

    /// Critical system state detected; restart required
    CriticalState { context: String, details: String },
}

impl fmt::Display for DreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DreamError::DimensionMismatch {
                expected,
                got,
                context,
            } => {
                write!(
                    f,
                    "Dimension mismatch in {}: expected {} dimensions, got {}",
                    context, expected, got
                )
            }
            DreamError::IndexNotBuilt { operation } => {
                write!(
                    f,
                    "Index not built: operation '{}' requires index to be built first. Call build() before querying.",
                    operation
                )
            }
            DreamError::CapacityExceeded { current, max } => {
                write!(
                    f,
                    "Pool capacity exceeded: current size {} exceeds maximum {}",
                    current, max
                )
            }
            DreamError::InvalidConfiguration {
                parameter,
                value,
                reason,
            } => {
                write!(
                    f,
                    "Invalid configuration for parameter '{}' with value '{}': {}",
                    parameter, value, reason
                )
            }
            DreamError::EmptyCollection { collection } => {
                write!(f, "Empty collection: {}", collection)
            }
            DreamError::InvalidParameter {
                parameter,
                value,
                constraint,
            } => {
                write!(
                    f,
                    "Invalid parameter '{}' = '{}': must satisfy {}",
                    parameter, value, constraint
                )
            }
            DreamError::IndexCorrupted { details } => {
                write!(f, "Index corrupted: {}", details)
            }
            DreamError::MemoryExceeded {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Memory budget exceeded: requested {} bytes, only {} available",
                    requested, available
                )
            }
            DreamError::FeatureUnavailable { feature, reason } => {
                write!(f, "Feature '{}' unavailable: {}", feature, reason)
            }
            DreamError::CriticalState { context, details } => {
                write!(
                    f,
                    "Critical state encountered in {}: {}. Manual intervention required.",
                    context, details
                )
            }
        }
    }
}

impl std::error::Error for DreamError {}

// Convenience constructors for common error patterns
impl DreamError {
    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, got: usize, context: impl Into<String>) -> Self {
        DreamError::DimensionMismatch {
            expected,
            got,
            context: context.into(),
        }
    }

    /// Create an index not built error
    pub fn index_not_built(operation: impl Into<String>) -> Self {
        DreamError::IndexNotBuilt {
            operation: operation.into(),
        }
    }

    /// Create a capacity exceeded error
    pub fn capacity_exceeded(current: usize, max: usize) -> Self {
        DreamError::CapacityExceeded { current, max }
    }

    /// Create an invalid configuration error
    pub fn invalid_config(
        parameter: impl Into<String>,
        value: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        DreamError::InvalidConfiguration {
            parameter: parameter.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }

    /// Create an empty collection error
    pub fn empty_collection(collection: impl Into<String>) -> Self {
        DreamError::EmptyCollection {
            collection: collection.into(),
        }
    }

    /// Create an invalid parameter error
    pub fn invalid_parameter(
        parameter: impl Into<String>,
        value: impl Into<String>,
        constraint: impl Into<String>,
    ) -> Self {
        DreamError::InvalidParameter {
            parameter: parameter.into(),
            value: value.into(),
            constraint: constraint.into(),
        }
    }

    /// Create an index corrupted error
    pub fn index_corrupted(details: impl Into<String>) -> Self {
        DreamError::IndexCorrupted {
            details: details.into(),
        }
    }

    /// Create a memory exceeded error
    pub fn memory_exceeded(requested: usize, available: usize) -> Self {
        DreamError::MemoryExceeded {
            requested,
            available,
        }
    }

    /// Create a feature unavailable error
    pub fn feature_unavailable(feature: impl Into<String>, reason: impl Into<String>) -> Self {
        DreamError::FeatureUnavailable {
            feature: feature.into(),
            reason: reason.into(),
        }
    }

    /// Create a critical state error, signaling unrecoverable conditions
    pub fn critical_state(context: impl Into<String>, details: impl Into<String>) -> Self {
        DreamError::CriticalState {
            context: context.into(),
            details: details.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch_display() {
        let err = DreamError::dimension_mismatch(64, 32, "query embedding");
        let msg = err.to_string();
        assert!(msg.contains("64"));
        assert!(msg.contains("32"));
        assert!(msg.contains("query embedding"));
    }

    #[test]
    fn test_index_not_built_display() {
        let err = DreamError::index_not_built("search");
        let msg = err.to_string();
        assert!(msg.contains("search"));
        assert!(msg.contains("build()"));
    }

    #[test]
    fn test_capacity_exceeded_display() {
        let err = DreamError::capacity_exceeded(1000, 500);
        let msg = err.to_string();
        assert!(msg.contains("1000"));
        assert!(msg.contains("500"));
    }

    #[test]
    fn test_invalid_config_display() {
        let err = DreamError::invalid_config("k", "0", "must be > 0");
        let msg = err.to_string();
        assert!(msg.contains("k"));
        assert!(msg.contains("0"));
        assert!(msg.contains("must be > 0"));
    }

    #[test]
    fn test_empty_collection_display() {
        let err = DreamError::empty_collection("candidates");
        let msg = err.to_string();
        assert!(msg.contains("candidates"));
    }

    #[test]
    fn test_error_equality() {
        let err1 = DreamError::dimension_mismatch(64, 32, "test");
        let err2 = DreamError::dimension_mismatch(64, 32, "test");
        let err3 = DreamError::dimension_mismatch(64, 16, "test");

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DreamError>();
    }

    #[test]
    fn test_critical_state_display() {
        let err = DreamError::critical_state("unit test", "fatal");
        let msg = err.to_string();
        assert!(msg.contains("Critical state"));
        assert!(msg.contains("unit test"));
        assert!(msg.contains("fatal"));
    }
}
