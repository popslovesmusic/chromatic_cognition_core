use std::fmt::{self, Display};
use std::ops::{Add, Sub};

use ndarray::{Array3, Array4, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A 4-dimensional chromatic tensor representing an RGB color field with certainty weights.
///
/// The tensor structure is `[rows, cols, layers, 3]` where each cell contains:
/// - RGB color values (3 channels) in range [0.0, 1.0]
/// - A certainty/confidence weight ρ in range [0.0, 1.0]
///
/// # Examples
///
/// ```
/// use chromatic_cognition_core::ChromaticTensor;
///
/// // Create a deterministic random tensor
/// let tensor = ChromaticTensor::from_seed(42, 64, 64, 8);
///
/// // Get statistics
/// let stats = tensor.statistics();
/// println!("Mean RGB: {:?}", stats.mean_rgb);
/// println!("Variance: {}", stats.variance);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChromaticTensor {
    /// RGB color values as a 4D array: [rows, cols, layers, 3]
    pub colors: Array4<f32>,
    /// Certainty weights as a 3D array: [rows, cols, layers]
    pub certainty: Array3<f32>,
}

impl ChromaticTensor {
    /// Creates a new chromatic tensor initialized with zeros.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows in the tensor
    /// * `cols` - Number of columns in the tensor
    /// * `layers` - Number of depth layers in the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use chromatic_cognition_core::ChromaticTensor;
    ///
    /// let tensor = ChromaticTensor::new(64, 64, 8);
    /// assert_eq!(tensor.shape(), (64, 64, 8, 3));
    /// ```
    pub fn new(rows: usize, cols: usize, layers: usize) -> Self {
        Self {
            colors: Array4::zeros((rows, cols, layers, 3)),
            certainty: Array3::zeros((rows, cols, layers)),
        }
    }

    /// Creates a chromatic tensor from existing color and certainty arrays.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `colors` and `certainty` don't match.
    ///
    /// # Arguments
    ///
    /// * `colors` - 4D array of RGB values [rows, cols, layers, 3]
    /// * `certainty` - 3D array of certainty weights [rows, cols, layers]
    pub fn from_arrays(colors: Array4<f32>, certainty: Array3<f32>) -> Self {
        assert_eq!(colors.dim().0, certainty.dim().0);
        assert_eq!(colors.dim().1, certainty.dim().1);
        assert_eq!(colors.dim().2, certainty.dim().2);
        Self { colors, certainty }
    }

    /// Creates a deterministic random chromatic tensor from a seed value.
    ///
    /// Uses a linear congruential generator (LCG) for deterministic randomness,
    /// ensuring the same seed always produces the same tensor.
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed (use 0 for default seed of 1)
    /// * `rows` - Number of rows in the tensor
    /// * `cols` - Number of columns in the tensor
    /// * `layers` - Number of depth layers in the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use chromatic_cognition_core::ChromaticTensor;
    ///
    /// // Create two tensors with the same seed
    /// let tensor1 = ChromaticTensor::from_seed(42, 32, 32, 4);
    /// let tensor2 = ChromaticTensor::from_seed(42, 32, 32, 4);
    ///
    /// // They will be identical
    /// assert_eq!(tensor1.colors, tensor2.colors);
    /// ```
    pub fn from_seed(seed: u64, rows: usize, cols: usize, layers: usize) -> Self {
        let mut tensor = Self::new(rows, cols, layers);
        let state = if seed == 0 { 1 } else { seed };
        let total = rows * cols * layers;

        tensor
            .colors
            .as_slice_mut()
            .expect("ndarray uses contiguous layout")
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, value)| {
                let step = idx as u64 + state;
                let next = lcg(step);
                *value = normalized(next);
            });

        tensor
            .certainty
            .as_slice_mut()
            .expect("ndarray uses contiguous layout")
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, value)| {
                let step = idx as u64 + state.wrapping_add(total as u64);
                let next = lcg(step);
                *value = normalized(next.wrapping_mul(3)).mul_add(0.9, 0.1);
            });

        tensor
    }

    pub fn shape(&self) -> (usize, usize, usize, usize) {
        self.colors.dim()
    }

    pub fn normalize(&self) -> Self {
        let mut colors = self.colors.clone();
        colors
            .as_slice_mut()
            .expect("contiguous")
            .par_iter_mut()
            .for_each(|value| {
                if !value.is_finite() {
                    *value = 0.0;
                }
            });
        let max_value = colors
            .as_slice()
            .expect("contiguous")
            .par_iter()
            .cloned()
            .reduce(|| 0.0, f32::max);
        if max_value > 1.0 {
            colors
                .as_slice_mut()
                .expect("contiguous")
                .par_iter_mut()
                .for_each(|value| *value /= max_value);
        }
        Self::from_arrays(colors, self.certainty.clone())
    }

    pub fn clamp(&self, min: f32, max: f32) -> Self {
        let mut colors = self.colors.clone();
        colors
            .as_slice_mut()
            .expect("contiguous")
            .par_iter_mut()
            .for_each(|value| *value = value.clamp(min, max));
        Self::from_arrays(colors, self.certainty.clone())
    }

    pub fn complement(&self) -> Self {
        let mut colors = self.colors.clone();
        let (rows, cols, layers, _) = colors.dim();
        for row in 0..rows {
            for col in 0..cols {
                for layer in 0..layers {
                    let g = colors[[row, col, layer, 1]];
                    let b = colors[[row, col, layer, 2]];
                    colors[[row, col, layer, 1]] = 1.0 - g;
                    colors[[row, col, layer, 2]] = 1.0 - b;
                }
            }
        }
        Self::from_arrays(colors, self.certainty.clone())
    }

    pub fn saturate(&self, alpha: f32) -> Self {
        let mut colors = self.colors.clone();
        let (rows, cols, layers, _) = colors.dim();
        for row in 0..rows {
            for col in 0..cols {
                for layer in 0..layers {
                    let r = colors[[row, col, layer, 0]];
                    let g = colors[[row, col, layer, 1]];
                    let b = colors[[row, col, layer, 2]];
                    let mean = (r + g + b) / 3.0;
                    colors[[row, col, layer, 0]] = mean + (r - mean) * alpha;
                    colors[[row, col, layer, 1]] = mean + (g - mean) * alpha;
                    colors[[row, col, layer, 2]] = mean + (b - mean) * alpha;
                }
            }
        }
        Self::from_arrays(colors, self.certainty.clone())
    }

    pub fn statistics(&self) -> TensorStatistics {
        let (rows, cols, layers, _) = self.colors.dim();
        let cells = (rows * cols * layers) as f32;

        let mut mean_rgb = [0.0f32; 3];
        for channel in 0..3 {
            let channel_view = self.colors.index_axis(Axis(3), channel);
            let sum = if let Some(slice) = channel_view.as_slice() {
                slice.par_iter().cloned().sum::<f32>()
            } else {
                channel_view.iter().cloned().sum::<f32>()
            };
            mean_rgb[channel] = sum / cells;
        }

        let mut variance_sum = 0.0f32;
        for channel in 0..3 {
            let mean = mean_rgb[channel];
            let channel_view = self.colors.index_axis(Axis(3), channel);
            let variance = if let Some(slice) = channel_view.as_slice() {
                slice
                    .par_iter()
                    .map(|value| {
                        let diff = *value - mean;
                        diff * diff
                    })
                    .sum::<f32>()
            } else {
                channel_view
                    .iter()
                    .map(|value| {
                        let diff = *value - mean;
                        diff * diff
                    })
                    .sum::<f32>()
            };
            variance_sum += variance;
        }

        let variance = variance_sum / (cells * 3.0);
        let mean_certainty = if let Some(slice) = self.certainty.as_slice() {
            slice.par_iter().cloned().sum::<f32>() / cells
        } else {
            self.certainty.iter().cloned().sum::<f32>() / cells
        };

        TensorStatistics {
            mean_rgb,
            variance,
            mean_certainty,
        }
    }

    /// Get RGB values at a specific cell
    ///
    /// # Arguments
    /// * `row` - Row index
    /// * `col` - Column index
    /// * `layer` - Layer index
    ///
    /// # Returns
    /// Array of [r, g, b] values
    pub fn get_rgb(&self, row: usize, col: usize, layer: usize) -> [f32; 3] {
        [
            self.colors[[row, col, layer, 0]],
            self.colors[[row, col, layer, 1]],
            self.colors[[row, col, layer, 2]],
        ]
    }

    /// Get tensor dimensions (rows, cols, layers)
    pub fn dims(&self) -> (usize, usize, usize) {
        let (rows, cols, layers, _) = self.colors.dim();
        (rows, cols, layers)
    }

    /// Get mean RGB values across all cells
    pub fn mean_rgb(&self) -> [f32; 3] {
        self.statistics().mean_rgb
    }

    /// Get total number of cells in tensor
    pub fn total_cells(&self) -> usize {
        let (rows, cols, layers) = self.dims();
        rows * cols * layers
    }

    /// Get number of rows
    pub fn rows(&self) -> usize {
        self.colors.dim().0
    }

    /// Get number of columns
    pub fn cols(&self) -> usize {
        self.colors.dim().1
    }

    /// Get number of layers
    pub fn layers(&self) -> usize {
        self.colors.dim().2
    }
}

impl Display for ChromaticTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.statistics();
        write!(
            f,
            "ChromaticTensor {}x{}x{} mean_rgb=({:.3},{:.3},{:.3}) variance={:.5}",
            self.colors.dim().0,
            self.colors.dim().1,
            self.colors.dim().2,
            stats.mean_rgb[0],
            stats.mean_rgb[1],
            stats.mean_rgb[2],
            stats.variance,
        )
    }
}

impl Add for ChromaticTensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.colors.dim(),
            rhs.colors.dim(),
            "tensor shapes must match"
        );
        let colors = &self.colors + &rhs.colors;
        let certainty = (&self.certainty + &rhs.certainty) * 0.5;
        Self::from_arrays(colors, certainty)
    }
}

impl Sub for ChromaticTensor {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.colors.dim(),
            rhs.colors.dim(),
            "tensor shapes must match"
        );
        let colors = &self.colors - &rhs.colors;
        let certainty = (&self.certainty + &rhs.certainty) * 0.5;
        Self::from_arrays(colors, certainty)
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct TensorStatistics {
    pub mean_rgb: [f32; 3],
    pub variance: f32,
    pub mean_certainty: f32,
}

fn lcg(seed: u64) -> u64 {
    seed.wrapping_mul(1664525).wrapping_add(1013904223)
}

fn normalized(value: u64) -> f32 {
    let fraction = (value & 0xFFFF_FFFF) as f32 / (u32::MAX as f32);
    fraction.clamp(0.0, 1.0)
}
