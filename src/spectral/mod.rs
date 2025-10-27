//! Spectral analysis module for chromatic tensors.
//!
//! Provides FFT-based feature extraction and frequency-domain analysis
//! for chromatic fields. Used by the Learner to compute spectral entropy
//! and identify frequency patterns in dream tensors.

pub mod fft;

pub use fft::{
    compute_spectral_entropy, extract_spectral_features, SpectralFeatures, WindowFunction,
};
