//! Continuous embedding mapper for semantic dream retrieval.
//!
//! This module implements Phase 4: Continuous Embedding / Soft Indexing.
//! Instead of hard class-based retrieval, dreams are mapped to a continuous
//! latent space and retrieved by semantic similarity (cosine/euclidean).
//!
//! **Key Concept:** Fuse multiple features into a fixed-dimension embedding:
//! - Chromatic signature (RGB): 3 dims
//! - Spectral features (entropy, band energies): ~6 dims
//! - Class one-hot (optional): 10 dims
//! - Utility priors: 1-2 dims
//!
//! Output: Fixed D-dimensional vector (default D=64) for ANN retrieval.

use crate::data::ColorClass;
use crate::dream::bias::BiasProfile;
use crate::dream::simple_pool::DreamEntry;
use crate::spectral::SpectralFeatures;

/// Query signature for embedding-based retrieval.
#[derive(Debug, Clone)]
pub struct QuerySignature {
    /// Chromatic RGB signature
    pub chroma: [f32; 3],

    /// Optional class hint for class-conditional retrieval
    pub class_hint: Option<ColorClass>,

    /// Optional spectral features
    pub spectral: Option<SpectralFeatures>,

    /// Optional utility prior (e.g., mean utility from feedback)
    pub utility_prior: Option<f32>,
}

impl QuerySignature {
    /// Create a simple query from RGB signature.
    pub fn from_chroma(chroma: [f32; 3]) -> Self {
        Self {
            chroma,
            class_hint: None,
            spectral: None,
            utility_prior: None,
        }
    }

    /// Create query with class hint.
    pub fn with_class(chroma: [f32; 3], class: ColorClass) -> Self {
        Self {
            chroma,
            class_hint: Some(class),
            spectral: None,
            utility_prior: None,
        }
    }
}

/// Embedding mapper that fuses multiple features into fixed-dimension vectors.
///
/// **Architecture:** Linear projection with layer normalization
/// ```text
/// features = [rgb(3), spectral(6), class_onehot(10), utility(1)]
/// z = LayerNorm(features · W + b)
/// ```
///
/// Currently uses a simple deterministic linear mapping (no learned weights).
pub struct EmbeddingMapper {
    /// Output embedding dimension
    pub dim: usize,

    /// Whether to include class one-hot encoding
    pub include_class: bool,

    /// Whether to include spectral features
    pub include_spectral: bool,

    /// Whether to include utility features
    pub include_utility: bool,
}

impl EmbeddingMapper {
    /// Create a new embedding mapper.
    ///
    /// # Arguments
    /// * `dim` - Output embedding dimension (e.g., 64)
    ///
    /// # Returns
    /// * EmbeddingMapper with default feature configuration
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            include_class: true,
            include_spectral: true,
            include_utility: true,
        }
    }

    /// Encode a dream entry into an embedding vector.
    ///
    /// # Arguments
    /// * `entry` - Dream entry to encode
    /// * `bias` - Optional bias profile for conditioning
    ///
    /// # Returns
    /// * D-dimensional embedding vector
    pub fn encode_entry(&self, entry: &DreamEntry, bias: Option<&BiasProfile>) -> Vec<f32> {
        let mut features = Vec::new();

        // 1. Chromatic signature (3 dims)
        features.extend_from_slice(&entry.chroma_signature);

        // 2. Spectral features (6 dims) - Always present since Phase 4 optimization
        if self.include_spectral {
            let spectral = &entry.spectral_features;
            features.push(spectral.entropy);
            features.push(spectral.low_freq_energy);
            features.push(spectral.mid_freq_energy);
            features.push(spectral.high_freq_energy);
            features.push(spectral.mean_psd);

            // Dominant frequency (averaged across RGB)
            let dom_freq_mean = (spectral.dominant_frequencies[0] as f32
                + spectral.dominant_frequencies[1] as f32
                + spectral.dominant_frequencies[2] as f32)
                / 3.0;
            features.push(dom_freq_mean);
        }

        // 3. Class one-hot (10 dims)
        if self.include_class {
            if let Some(class) = entry.class_label {
                let onehot = self.class_to_onehot(class);
                features.extend_from_slice(&onehot);
            } else {
                // Unknown class - use uniform distribution
                features.extend_from_slice(&[0.1; 10]);
            }
        }

        // 4. Utility features (2 dims: raw utility + bias weight)
        if self.include_utility {
            let utility = entry.utility.unwrap_or(0.0);
            features.push(utility);

            // Bias weight if available
            if let Some(bias_profile) = bias {
                if let Some(class) = entry.class_label {
                    let weight = bias_profile.class_weight(class);
                    features.push(weight);
                } else {
                    features.push(0.5); // Neutral weight
                }
            } else {
                features.push(0.5);
            }
        }

        // 5. Project to target dimension and normalize
        self.project_and_normalize(features)
    }

    /// Encode a query signature into an embedding vector.
    ///
    /// # Arguments
    /// * `query` - Query signature
    /// * `bias` - Optional bias profile for conditioning
    ///
    /// # Returns
    /// * D-dimensional embedding vector
    pub fn encode_query(&self, query: &QuerySignature, bias: Option<&BiasProfile>) -> Vec<f32> {
        let mut features = Vec::new();

        // 1. Chromatic signature (3 dims)
        features.extend_from_slice(&query.chroma);

        // 2. Spectral features (6 dims)
        if self.include_spectral {
            if let Some(ref spectral) = query.spectral {
                features.push(spectral.entropy);
                features.push(spectral.low_freq_energy);
                features.push(spectral.mid_freq_energy);
                features.push(spectral.high_freq_energy);
                features.push(spectral.mean_psd);

                let dom_freq_mean = (spectral.dominant_frequencies[0] as f32
                    + spectral.dominant_frequencies[1] as f32
                    + spectral.dominant_frequencies[2] as f32)
                    / 3.0;
                features.push(dom_freq_mean);
            } else {
                features.extend_from_slice(&[0.0; 6]);
            }
        }

        // 3. Class one-hot (10 dims)
        if self.include_class {
            if let Some(class) = query.class_hint {
                let onehot = self.class_to_onehot(class);
                features.extend_from_slice(&onehot);
            } else {
                // No class hint - use uniform
                features.extend_from_slice(&[0.1; 10]);
            }
        }

        // 4. Utility features (2 dims)
        if self.include_utility {
            let utility_prior = query.utility_prior.unwrap_or(0.0);
            features.push(utility_prior);

            // Bias weight
            if let Some(bias_profile) = bias {
                if let Some(class) = query.class_hint {
                    let weight = bias_profile.class_weight(class);
                    features.push(weight);
                } else {
                    features.push(0.5);
                }
            } else {
                features.push(0.5);
            }
        }

        self.project_and_normalize(features)
    }

    /// Convert ColorClass to one-hot encoding.
    fn class_to_onehot(&self, class: ColorClass) -> [f32; 10] {
        let mut onehot = [0.0; 10];
        let idx = class as usize;
        onehot[idx] = 1.0;
        onehot
    }

    /// Project features to target dimension and apply layer normalization.
    ///
    /// Uses a simple deterministic projection (no learned weights).
    fn project_and_normalize(&self, features: Vec<f32>) -> Vec<f32> {
        let feature_dim = features.len();

        // Simple linear projection: repeat/truncate to target dim
        let mut projected = Vec::with_capacity(self.dim);

        if self.dim <= feature_dim {
            // Truncate
            projected.extend_from_slice(&features[..self.dim]);
        } else {
            // Repeat features cyclically to fill dimension
            for i in 0..self.dim {
                projected.push(features[i % feature_dim]);
            }
        }

        // Layer normalization: (x - mean) / std
        let mean = projected.iter().sum::<f32>() / projected.len() as f32;
        let variance =
            projected.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / projected.len() as f32;
        let std = (variance + 1e-8).sqrt(); // Add epsilon for numerical stability

        projected.iter().map(|&x| (x - mean) / std).collect()
    }

    /// Get expected feature dimension (before projection).
    pub fn feature_dim(&self) -> usize {
        let mut dim = 3; // RGB

        if self.include_spectral {
            dim += 6; // entropy + 3 band energies + mean_psd + dom_freq
        }

        if self.include_class {
            dim += 10; // one-hot
        }

        if self.include_utility {
            dim += 2; // utility + bias_weight
        }

        dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SolverResult;
    use crate::tensor::ChromaticTensor;
    use serde_json::json;

    fn make_dream_entry() -> DreamEntry {
        let tensor = ChromaticTensor::new(8, 8, 4);
        let result = SolverResult {
            energy: 0.1,
            coherence: 0.8,
            violation: 0.0,
            grad: None,
            mask: None,
            meta: json!({}),
        };

        let mut entry = DreamEntry::new(tensor, result);
        entry.chroma_signature = [1.0, 0.5, 0.2];
        entry.class_label = Some(ColorClass::Red);
        entry.utility = Some(0.3);
        entry
    }

    #[test]
    fn test_embedding_mapper_creation() {
        let mapper = EmbeddingMapper::new(64);
        assert_eq!(mapper.dim, 64);
        assert!(mapper.include_class);
        assert!(mapper.include_spectral);
        assert!(mapper.include_utility);
    }

    #[test]
    fn test_feature_dim() {
        let mapper = EmbeddingMapper::new(64);
        // RGB(3) + spectral(6) + class(10) + utility(2) = 21
        assert_eq!(mapper.feature_dim(), 21);
    }

    #[test]
    fn test_encode_entry_shape() {
        let mapper = EmbeddingMapper::new(64);
        let entry = make_dream_entry();

        let embedding = mapper.encode_entry(&entry, None);
        assert_eq!(embedding.len(), 64);
    }

    #[test]
    fn test_encode_entry_deterministic() {
        let mapper = EmbeddingMapper::new(64);
        let entry = make_dream_entry();

        let embed1 = mapper.encode_entry(&entry, None);
        let embed2 = mapper.encode_entry(&entry, None);

        assert_eq!(embed1, embed2);
    }

    #[test]
    fn test_encode_query_shape() {
        let mapper = EmbeddingMapper::new(64);
        let query = QuerySignature::from_chroma([1.0, 0.0, 0.0]);

        let embedding = mapper.encode_query(&query, None);
        assert_eq!(embedding.len(), 64);
    }

    #[test]
    fn test_class_to_onehot() {
        let mapper = EmbeddingMapper::new(64);
        let onehot = mapper.class_to_onehot(ColorClass::Red);

        assert_eq!(onehot[0], 1.0); // Red is index 0
        assert_eq!(onehot[1], 0.0);
        assert_eq!(onehot.iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn test_layer_normalization() {
        let mapper = EmbeddingMapper::new(64);
        let entry = make_dream_entry();

        let embedding = mapper.encode_entry(&entry, None);

        // Check layer norm properties: mean ≈ 0, std ≈ 1
        let mean = embedding.iter().sum::<f32>() / embedding.len() as f32;
        let variance =
            embedding.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / embedding.len() as f32;
        let std = variance.sqrt();

        assert!(mean.abs() < 0.01); // Mean should be ~0
        assert!((std - 1.0).abs() < 0.01); // Std should be ~1
    }

    #[test]
    fn test_different_classes_different_embeddings() {
        let mapper = EmbeddingMapper::new(64);

        let mut entry1 = make_dream_entry();
        entry1.class_label = Some(ColorClass::Red);

        let mut entry2 = make_dream_entry();
        entry2.class_label = Some(ColorClass::Blue);

        let embed1 = mapper.encode_entry(&entry1, None);
        let embed2 = mapper.encode_entry(&entry2, None);

        assert_ne!(embed1, embed2);
    }

    #[test]
    fn test_query_with_class_hint() {
        let mapper = EmbeddingMapper::new(64);

        let query1 = QuerySignature::from_chroma([1.0, 0.0, 0.0]);
        let query2 = QuerySignature::with_class([1.0, 0.0, 0.0], ColorClass::Red);

        let embed1 = mapper.encode_query(&query1, None);
        let embed2 = mapper.encode_query(&query2, None);

        // Different due to class hint
        assert_ne!(embed1, embed2);
    }
}
