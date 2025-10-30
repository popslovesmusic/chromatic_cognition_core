//! Color classification dataset for validation experiments
//!
//! Provides a simple 10-class color classification task with synthetic data.
//! Each sample is a ChromaticTensor filled with a specific color + noise.

use crate::tensor::ChromaticTensor;
use ndarray::Array4;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// 10 standard color classes for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(usize)]
pub enum ColorClass {
    Red = 0,
    Green = 1,
    Blue = 2,
    Yellow = 3,
    Cyan = 4,
    Magenta = 5,
    Orange = 6,
    Purple = 7,
    White = 8,
    Black = 9,
}

impl ColorClass {
    /// Get the canonical RGB values for this color class
    pub fn rgb(&self) -> [f32; 3] {
        match self {
            ColorClass::Red => [1.0, 0.0, 0.0],
            ColorClass::Green => [0.0, 1.0, 0.0],
            ColorClass::Blue => [0.0, 0.0, 1.0],
            ColorClass::Yellow => [1.0, 1.0, 0.0],
            ColorClass::Cyan => [0.0, 1.0, 1.0],
            ColorClass::Magenta => [1.0, 0.0, 1.0],
            ColorClass::Orange => [1.0, 0.5, 0.0],
            ColorClass::Purple => [0.5, 0.0, 0.5],
            ColorClass::White => [1.0, 1.0, 1.0],
            ColorClass::Black => [0.0, 0.0, 0.0],
        }
    }

    /// Get color class from index (0-9)
    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(ColorClass::Red),
            1 => Some(ColorClass::Green),
            2 => Some(ColorClass::Blue),
            3 => Some(ColorClass::Yellow),
            4 => Some(ColorClass::Cyan),
            5 => Some(ColorClass::Magenta),
            6 => Some(ColorClass::Orange),
            7 => Some(ColorClass::Purple),
            8 => Some(ColorClass::White),
            9 => Some(ColorClass::Black),
            _ => None,
        }
    }

    /// Total number of color classes
    pub fn num_classes() -> usize {
        10
    }

    /// Get all color classes
    pub fn all() -> [ColorClass; 10] {
        [
            ColorClass::Red,
            ColorClass::Green,
            ColorClass::Blue,
            ColorClass::Yellow,
            ColorClass::Cyan,
            ColorClass::Magenta,
            ColorClass::Orange,
            ColorClass::Purple,
            ColorClass::White,
            ColorClass::Black,
        ]
    }
}

/// A single training/validation sample
#[derive(Clone, Serialize, Deserialize)]
pub struct ColorSample {
    pub tensor: ChromaticTensor,
    pub label: ColorClass,
}

/// Configuration for dataset generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Tensor dimensions (rows, cols, layers)
    pub tensor_size: (usize, usize, usize),
    /// Amount of Gaussian noise to add (stddev)
    pub noise_level: f32,
    /// Number of samples per color class
    pub samples_per_class: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            tensor_size: (16, 16, 4),
            noise_level: 0.1,
            samples_per_class: 100,
            seed: 42,
        }
    }
}

/// Color classification dataset
pub struct ColorDataset {
    pub samples: Vec<ColorSample>,
    pub config: DatasetConfig,
}

impl ColorDataset {
    /// Generate a new synthetic color dataset
    pub fn generate(config: DatasetConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut samples = Vec::new();

        let (rows, cols, layers) = config.tensor_size;

        for color_class in ColorClass::all() {
            let base_rgb = color_class.rgb();

            for _ in 0..config.samples_per_class {
                // Create tensor filled with base color + noise
                let mut colors = Array4::zeros((rows, cols, layers, 3));
                let mut certainty = ndarray::Array3::zeros((rows, cols, layers));

                for r in 0..rows {
                    for c in 0..cols {
                        for l in 0..layers {
                            // Base color + Gaussian noise
                            for channel in 0..3 {
                                let noise = rng.gen::<f32>() * config.noise_level * 2.0
                                    - config.noise_level;
                                let value = (base_rgb[channel] + noise).clamp(0.0, 1.0);
                                colors[[r, c, l, channel]] = value;
                            }

                            // Random certainty
                            certainty[[r, c, l]] = rng.gen::<f32>() * 0.5 + 0.5;
                        }
                    }
                }

                let tensor = ChromaticTensor::from_arrays(colors, certainty);
                samples.push(ColorSample {
                    tensor,
                    label: color_class,
                });
            }
        }

        // Shuffle samples
        use rand::seq::SliceRandom;
        samples.shuffle(&mut rng);

        Self { samples, config }
    }

    /// Split dataset into train and validation sets
    ///
    /// # Arguments
    /// * `train_ratio` - Fraction of data to use for training (e.g., 0.8)
    ///
    /// # Returns
    /// Tuple of (train_dataset, val_dataset)
    pub fn split(self, train_ratio: f32) -> (Vec<ColorSample>, Vec<ColorSample>) {
        let split_idx = (self.samples.len() as f32 * train_ratio) as usize;
        let (train, val) = self.samples.split_at(split_idx);
        (train.to_vec(), val.to_vec())
    }

    /// Get total number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get samples by batch
    pub fn batch(&self, batch_size: usize, batch_idx: usize) -> &[ColorSample] {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(self.samples.len());
        &self.samples[start..end]
    }

    /// Get number of batches for given batch size
    pub fn num_batches(&self, batch_size: usize) -> usize {
        (self.samples.len() + batch_size - 1) / batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_class_rgb() {
        assert_eq!(ColorClass::Red.rgb(), [1.0, 0.0, 0.0]);
        assert_eq!(ColorClass::Green.rgb(), [0.0, 1.0, 0.0]);
        assert_eq!(ColorClass::Blue.rgb(), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_color_class_from_index() {
        assert_eq!(ColorClass::from_index(0), Some(ColorClass::Red));
        assert_eq!(ColorClass::from_index(9), Some(ColorClass::Black));
        assert_eq!(ColorClass::from_index(10), None);
    }

    #[test]
    fn test_dataset_generation() {
        let config = DatasetConfig {
            tensor_size: (8, 8, 2),
            noise_level: 0.05,
            samples_per_class: 10,
            seed: 42,
        };

        let dataset = ColorDataset::generate(config);
        assert_eq!(dataset.len(), 100); // 10 classes * 10 samples

        // Check that we have samples from different classes
        let labels: Vec<_> = dataset.samples.iter().map(|s| s.label).collect();
        assert!(labels.contains(&ColorClass::Red));
        assert!(labels.contains(&ColorClass::Blue));
    }

    #[test]
    fn test_dataset_split() {
        let config = DatasetConfig {
            samples_per_class: 10,
            ..Default::default()
        };

        let dataset = ColorDataset::generate(config);
        let total_len = dataset.len();

        let (train, val) = dataset.split(0.8);
        assert_eq!(train.len() + val.len(), total_len);
        assert!(train.len() > val.len());
    }

    #[test]
    fn test_dataset_batching() {
        let config = DatasetConfig {
            samples_per_class: 5,
            ..Default::default()
        };

        let dataset = ColorDataset::generate(config);
        let batch_size = 10;
        let num_batches = dataset.num_batches(batch_size);

        assert_eq!(num_batches, 5); // 50 samples / 10 batch_size = 5

        let batch = dataset.batch(batch_size, 0);
        assert_eq!(batch.len(), batch_size);
    }
}
