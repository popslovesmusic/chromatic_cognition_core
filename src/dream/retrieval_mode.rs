//! Retrieval mode configuration for Phase 4 training integration.
//!
//! Defines how the dream pool retrieval should operate during training.

use serde::{Serialize, Deserialize};

/// Retrieval mode for dream pool queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalMode {
    /// Hard retrieval (Phase 3B): Class-aware cosine similarity on RGB
    /// Uses only chromatic signatures and optional class filtering
    Hard,

    /// Soft retrieval (Phase 4): Continuous embedding with ANN
    /// Uses EmbeddingMapper + SoftIndex + hybrid scoring
    Soft,

    /// Hybrid mode: Combine hard and soft retrieval results
    /// Retrieves from both methods and merges with deduplication
    Hybrid,
}

impl Default for RetrievalMode {
    fn default() -> Self {
        RetrievalMode::Hard
    }
}

impl RetrievalMode {
    /// Check if this mode requires soft index (Phase 4)
    pub fn requires_soft_index(&self) -> bool {
        matches!(self, RetrievalMode::Soft | RetrievalMode::Hybrid)
    }

    /// Check if this mode uses hard retrieval (Phase 3B)
    pub fn uses_hard_retrieval(&self) -> bool {
        matches!(self, RetrievalMode::Hard | RetrievalMode::Hybrid)
    }

    /// Check if this mode uses soft retrieval (Phase 4)
    pub fn uses_soft_retrieval(&self) -> bool {
        matches!(self, RetrievalMode::Soft | RetrievalMode::Hybrid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_mode_default() {
        let mode = RetrievalMode::default();
        assert_eq!(mode, RetrievalMode::Hard);
    }

    #[test]
    fn test_hard_mode_properties() {
        let mode = RetrievalMode::Hard;
        assert!(!mode.requires_soft_index());
        assert!(mode.uses_hard_retrieval());
        assert!(!mode.uses_soft_retrieval());
    }

    #[test]
    fn test_soft_mode_properties() {
        let mode = RetrievalMode::Soft;
        assert!(mode.requires_soft_index());
        assert!(!mode.uses_hard_retrieval());
        assert!(mode.uses_soft_retrieval());
    }

    #[test]
    fn test_hybrid_mode_properties() {
        let mode = RetrievalMode::Hybrid;
        assert!(mode.requires_soft_index());
        assert!(mode.uses_hard_retrieval());
        assert!(mode.uses_soft_retrieval());
    }
}
