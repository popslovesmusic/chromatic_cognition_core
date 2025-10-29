//! FFT-based feature extraction for chromatic tensors.
//!
//! This module implements frequency-domain analysis using Fast Fourier Transform (FFT)
//! to extract spectral features from chromatic fields. These features capture patterns
//! that may not be apparent in the spatial domain.
//!
//! **Use Case (Phase 3B):** The Learner can use spectral entropy and frequency features
//! to bias the Dreamer toward dreams with specific frequency characteristics that
//! historically helped training convergence.

use super::accumulate::{deterministic_mean, deterministic_sum};
use crate::tensor::ChromaticTensor;
use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

/// Window function for FFT preprocessing.
///
/// Windowing reduces spectral leakage by smoothly tapering the signal at edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowFunction {
    /// No windowing (rectangular)
    None,
    /// Hann window: 0.5 - 0.5*cos(2π*n/N)
    Hann,
    /// Hamming window: 0.54 - 0.46*cos(2π*n/N)
    Hamming,
}

impl WindowFunction {
    /// Generate window coefficients for a given length.
    ///
    /// # Arguments
    /// * `n` - Window length
    ///
    /// # Returns
    /// * Array of window coefficients [0, 1]
    pub fn generate(&self, n: usize) -> Vec<f32> {
        match self {
            WindowFunction::None => vec![1.0; n],
            WindowFunction::Hann => (0..n)
                .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
                .collect(),
            WindowFunction::Hamming => (0..n)
                .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / n as f32).cos())
                .collect(),
        }
    }
}

/// Spectral features extracted from a chromatic tensor.
///
/// Contains frequency-domain statistics computed via FFT.
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    /// Spectral entropy (0-1, higher = more complex frequency distribution)
    ///
    /// Measures the "disorder" in the frequency spectrum. High entropy indicates
    /// a broad, flat spectrum (complex patterns), while low entropy indicates
    /// a peaky spectrum (simple, periodic patterns).
    pub entropy: f32,

    /// Dominant frequency bin for each RGB channel
    ///
    /// Identifies the strongest frequency component in each color channel.
    /// Format: [red_freq_bin, green_freq_bin, blue_freq_bin]
    pub dominant_frequencies: [usize; 3],

    /// Energy in low frequency band (0-25% of Nyquist)
    ///
    /// Captures large-scale, smooth color variations.
    pub low_freq_energy: f32,

    /// Energy in mid frequency band (25-75% of Nyquist)
    ///
    /// Captures medium-scale patterns and textures.
    pub mid_freq_energy: f32,

    /// Energy in high frequency band (75-100% of Nyquist)
    ///
    /// Captures fine details and sharp transitions.
    pub high_freq_energy: f32,

    /// Mean power spectral density across all channels
    ///
    /// Average energy per frequency bin.
    pub mean_psd: f32,
}

/// Extract spectral features from a chromatic tensor.
///
/// Performs 2D FFT on each RGB channel and computes frequency-domain statistics.
///
/// # Arguments
/// * `tensor` - Input chromatic tensor (rows × cols × layers)
/// * `window` - Window function to apply before FFT
///
/// # Returns
/// * SpectralFeatures containing entropy, dominant frequencies, energy bands, etc.
///
/// # Example
/// ```
/// # use chromatic_cognition_core::tensor::ChromaticTensor;
/// # use chromatic_cognition_core::spectral::{extract_spectral_features, WindowFunction};
/// let tensor = ChromaticTensor::new(16, 16, 4);
/// let features = extract_spectral_features(&tensor, WindowFunction::Hann);
/// println!("Spectral entropy: {}", features.entropy);
/// ```
pub fn extract_spectral_features(
    tensor: &ChromaticTensor,
    window: WindowFunction,
) -> SpectralFeatures {
    // Extract RGB channels (average over layers)
    let red_channel = extract_channel(tensor, 0);
    let green_channel = extract_channel(tensor, 1);
    let blue_channel = extract_channel(tensor, 2);

    // Compute 2D FFT for each channel
    let red_spectrum = compute_2d_fft(&red_channel, window);
    let green_spectrum = compute_2d_fft(&green_channel, window);
    let blue_spectrum = compute_2d_fft(&blue_channel, window);

    // Compute power spectral densities
    let red_psd = compute_psd(&red_spectrum);
    let green_psd = compute_psd(&green_spectrum);
    let blue_psd = compute_psd(&blue_spectrum);

    // Find dominant frequencies
    let red_dom = find_dominant_frequency(&red_psd);
    let green_dom = find_dominant_frequency(&green_psd);
    let blue_dom = find_dominant_frequency(&blue_psd);

    // Compute frequency band energies (average across RGB)
    let (low_r, mid_r, high_r) = compute_band_energies(&red_psd);
    let (low_g, mid_g, high_g) = compute_band_energies(&green_psd);
    let (low_b, mid_b, high_b) = compute_band_energies(&blue_psd);

    let low_freq_energy = deterministic_mean(&[low_r, low_g, low_b]);
    let mid_freq_energy = deterministic_mean(&[mid_r, mid_g, mid_b]);
    let high_freq_energy = deterministic_mean(&[high_r, high_g, high_b]);

    // Compute spectral entropy (average across RGB)
    let entropy_r = compute_spectral_entropy(&red_psd);
    let entropy_g = compute_spectral_entropy(&green_psd);
    let entropy_b = compute_spectral_entropy(&blue_psd);
    let entropy = deterministic_mean(&[entropy_r, entropy_g, entropy_b]);

    // Mean PSD
    let mean_psd = deterministic_sum(&[
        deterministic_sum(&red_psd),
        deterministic_sum(&green_psd),
        deterministic_sum(&blue_psd),
    ]) / (3.0 * red_psd.len() as f32);

    SpectralFeatures {
        entropy,
        dominant_frequencies: [red_dom, green_dom, blue_dom],
        low_freq_energy,
        mid_freq_energy,
        high_freq_energy,
        mean_psd,
    }
}

/// Compute spectral entropy from a power spectral density.
///
/// **Entropy** = -Σ p_i log(p_i), where p_i = normalized PSD
///
/// Normalized to [0, 1] by dividing by log(N).
///
/// # Arguments
/// * `psd` - Power spectral density (non-negative values)
///
/// # Returns
/// * Spectral entropy in [0, 1]
///
/// # Example
/// ```
/// # use chromatic_cognition_core::spectral::compute_spectral_entropy;
/// let psd = vec![0.5, 0.3, 0.2]; // Example PSD
/// let entropy = compute_spectral_entropy(&psd);
/// assert!(entropy >= 0.0 && entropy <= 1.0);
/// ```
pub fn compute_spectral_entropy(psd: &[f32]) -> f32 {
    if psd.is_empty() {
        return 0.0;
    }

    // Normalize PSD to probability distribution
    let total = deterministic_sum(psd);
    if total < 1e-10 {
        return 0.0; // Flat spectrum (max entropy)
    }

    let probs: Vec<f32> = psd.iter().map(|&p| p / total).collect();

    // Compute Shannon entropy: -Σ p_i log(p_i)
    let mut terms = Vec::with_capacity(probs.len());
    for &p in &probs {
        if p > 1e-10 {
            terms.push(-p * p.ln());
        }
    }
    let entropy = deterministic_sum(&terms);

    // Normalize by max entropy (log(N))
    let max_entropy = (psd.len() as f32).ln();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

/// Extract a single color channel from tensor (averaged over layers).
fn extract_channel(tensor: &ChromaticTensor, channel: usize) -> Array2<f32> {
    let (rows, cols, layers, _) = tensor.shape();
    let mut result = Array2::zeros((rows, cols));

    let mut buffer = Vec::with_capacity(layers);
    for i in 0..rows {
        for j in 0..cols {
            buffer.clear();
            for k in 0..layers {
                buffer.push(tensor.get_rgb(i, j, k)[channel]);
            }
            result[[i, j]] = deterministic_mean(&buffer);
        }
    }

    result
}

/// Compute 2D FFT of a real-valued 2D array.
fn compute_2d_fft(data: &Array2<f32>, window: WindowFunction) -> Array2<Complex<f32>> {
    let (rows, cols) = data.dim();

    // Generate window for rows and cols
    let row_window = window.generate(rows);
    let col_window = window.generate(cols);

    // Apply 2D window (separable: outer product)
    let mut windowed = data.clone();
    for i in 0..rows {
        for j in 0..cols {
            windowed[[i, j]] *= row_window[i] * col_window[j];
        }
    }

    // Convert to complex
    let mut complex_data: Vec<Vec<Complex<f32>>> = windowed
        .outer_iter()
        .map(|row| row.iter().map(|&x| Complex::new(x, 0.0)).collect())
        .collect();

    // FFT planner
    let mut planner = FftPlanner::new();
    let fft_row = planner.plan_fft_forward(cols);
    let fft_col = planner.plan_fft_forward(rows);

    // FFT along rows
    for row in &mut complex_data {
        fft_row.process(row);
    }

    // Transpose for column FFT
    let mut transposed = vec![vec![Complex::new(0.0, 0.0); rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = complex_data[i][j];
        }
    }

    // FFT along columns (now rows of transposed)
    for col in &mut transposed {
        fft_col.process(col);
    }

    // Transpose back
    let mut result_vec = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            result_vec[i][j] = transposed[j][i];
        }
    }

    // Convert to Array2
    let flat: Vec<Complex<f32>> = result_vec.into_iter().flatten().collect();
    Array2::from_shape_vec((rows, cols), flat).unwrap()
}

/// Compute power spectral density from FFT output.
///
/// PSD[k] = |FFT[k]|^2
fn compute_psd(spectrum: &Array2<Complex<f32>>) -> Vec<f32> {
    spectrum
        .iter()
        .map(|c| c.norm_sqr()) // |c|^2 = real^2 + imag^2
        .collect()
}

/// Find the index of the dominant (maximum) frequency in PSD.
fn find_dominant_frequency(psd: &[f32]) -> usize {
    psd.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Compute energy in low/mid/high frequency bands.
///
/// Bands: [0-25%, 25-75%, 75-100%] of total frequency range
///
/// Returns: (low_energy, mid_energy, high_energy)
fn compute_band_energies(psd: &[f32]) -> (f32, f32, f32) {
    let n = psd.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }

    let low_cutoff = n / 4;
    let high_cutoff = 3 * n / 4;

    let low_energy = deterministic_sum(&psd[..low_cutoff]);
    let mid_energy = deterministic_sum(&psd[low_cutoff..high_cutoff]);
    let high_energy = deterministic_sum(&psd[high_cutoff..]);

    (low_energy, mid_energy, high_energy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_function_none() {
        let window = WindowFunction::None.generate(5);
        assert_eq!(window, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_window_function_hann() {
        let window = WindowFunction::Hann.generate(5);
        assert_eq!(window.len(), 5);
        // Hann window: 0.5 - 0.5*cos(2π*n/N)
        // At n=0: 0.5 - 0.5*cos(0) = 0.5 - 0.5 = 0.0
        // At n=4: 0.5 - 0.5*cos(8π/5) ≈ 0.095 (not exactly 0 for N=5)
        // The window is symmetric and peaks in the middle
        assert!(window[0] < 0.01); // Near zero at start
        assert!(window[2] > 0.9); // Peak near 1 in middle; ignore odd-length endpoints
    }

    #[test]
    fn test_spectral_entropy_uniform() {
        // Uniform distribution should have high entropy
        let psd = vec![1.0, 1.0, 1.0, 1.0];
        let entropy = compute_spectral_entropy(&psd);
        assert!((entropy - 1.0).abs() < 0.01); // Should be ~1.0
    }

    #[test]
    fn test_spectral_entropy_peaked() {
        // Single peak should have low entropy
        let psd = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let entropy = compute_spectral_entropy(&psd);
        assert!(entropy < 0.1); // Should be ~0.0
    }

    #[test]
    fn test_spectral_entropy_empty() {
        let psd = vec![];
        let entropy = compute_spectral_entropy(&psd);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_extract_spectral_features() {
        let tensor = ChromaticTensor::new(8, 8, 4);
        let features = extract_spectral_features(&tensor, WindowFunction::Hann);

        // Check all fields are computed
        assert!(features.entropy >= 0.0 && features.entropy <= 1.0);
        assert!(features.low_freq_energy >= 0.0);
        assert!(features.mid_freq_energy >= 0.0);
        assert!(features.high_freq_energy >= 0.0);
        assert!(features.mean_psd >= 0.0);
    }

    #[test]
    fn test_band_energies() {
        let psd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (low, mid, high) = compute_band_energies(&psd);

        // Low: [0, 1] = 1+2 = 3
        // Mid: [2, 3, 4, 5] = 3+4+5+6 = 18
        // High: [6, 7] = 7+8 = 15
        assert_eq!(low, 3.0);
        assert_eq!(mid, 18.0);
        assert_eq!(high, 15.0);
    }

    #[test]
    fn test_find_dominant_frequency() {
        let psd = vec![1.0, 5.0, 2.0, 3.0];
        let dom = find_dominant_frequency(&psd);
        assert_eq!(dom, 1); // Index of 5.0
    }

    #[test]
    fn test_extract_channel() {
        let tensor = ChromaticTensor::new(4, 4, 2);
        let red = extract_channel(&tensor, 0);
        assert_eq!(red.dim(), (4, 4));
    }
}
