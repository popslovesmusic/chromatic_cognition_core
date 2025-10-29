//! Spectral analysis module for chromatic tensors.
//!
//! Provides FFT-based feature extraction and frequency-domain analysis
//! for chromatic fields. Used by the Learner to compute spectral entropy
//! and identify frequency patterns in dream tensors.

pub mod accumulate;
pub mod bridge;
pub mod color;
pub mod fft;

pub use bridge::{canonical_hue, circular_mean, SpectralTensor};
pub use color::{delta_e94, srgb_to_lab, DELTA_E94_TOLERANCE};
pub use fft::{
    compute_spectral_entropy, extract_spectral_features, SpectralFeatures, WindowFunction,
};
