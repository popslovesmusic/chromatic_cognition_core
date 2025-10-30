//! Color pattern generation for training datasets.

use crate::tensor::ChromaticTensor;
use rand::{Rng, SeedableRng};

/// A labeled color pattern for classification.
#[derive(Clone, Debug)]
pub struct ColorPattern {
    /// The chromatic tensor
    pub tensor: ChromaticTensor,
    /// Class label (0 = red, 1 = green, 2 = blue for primary colors)
    pub label: usize,
    /// Human-readable description
    pub description: String,
}

/// Generates a dataset of primary color patterns.
///
/// Creates synthetic patterns dominated by red, green, or blue.
///
/// # Arguments
///
/// * `samples_per_class` - Number of samples to generate per class
/// * `rows` - Tensor height
/// * `cols` - Tensor width
/// * `layers` - Tensor depth
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// Vector of labeled color patterns
///
/// # Examples
///
/// ```
/// use chromatic_cognition_core::data::generate_primary_color_dataset;
///
/// let dataset = generate_primary_color_dataset(100, 16, 16, 4, 42);
/// assert_eq!(dataset.len(), 300); // 100 per class × 3 classes
/// ```
pub fn generate_primary_color_dataset(
    samples_per_class: usize,
    rows: usize,
    cols: usize,
    layers: usize,
    seed: u64,
) -> Vec<ColorPattern> {
    let mut dataset = Vec::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let class_names = ["red", "green", "blue"];
    let base_colors = [
        [0.9, 0.1, 0.1], // Red
        [0.1, 0.9, 0.1], // Green
        [0.1, 0.1, 0.9], // Blue
    ];

    for class in 0..3 {
        for sample_idx in 0..samples_per_class {
            let sample_seed = seed.wrapping_add((class * samples_per_class + sample_idx) as u64);

            // Start with random tensor
            let mut tensor = ChromaticTensor::from_seed(sample_seed, rows, cols, layers);

            // Bias toward class color
            let base_color = base_colors[class];
            let intensity = 0.7 + rng.gen::<f32>() * 0.3; // 0.7 to 1.0

            for row in 0..rows {
                for col in 0..cols {
                    for layer in 0..layers {
                        // Mix with base color
                        let r = tensor.colors[[row, col, layer, 0]];
                        let g = tensor.colors[[row, col, layer, 1]];
                        let b = tensor.colors[[row, col, layer, 2]];

                        tensor.colors[[row, col, layer, 0]] =
                            (r * (1.0 - intensity) + base_color[0] * intensity).clamp(0.0, 1.0);
                        tensor.colors[[row, col, layer, 1]] =
                            (g * (1.0 - intensity) + base_color[1] * intensity).clamp(0.0, 1.0);
                        tensor.colors[[row, col, layer, 2]] =
                            (b * (1.0 - intensity) + base_color[2] * intensity).clamp(0.0, 1.0);

                        // Add some certainty variation
                        tensor.certainty[[row, col, layer]] = 0.5 + rng.gen::<f32>() * 0.5;
                    }
                }
            }

            // Add noise
            let noise_level = 0.1;
            for row in 0..rows {
                for col in 0..cols {
                    for layer in 0..layers {
                        tensor.colors[[row, col, layer, 0]] +=
                            (rng.gen::<f32>() - 0.5) * noise_level;
                        tensor.colors[[row, col, layer, 1]] +=
                            (rng.gen::<f32>() - 0.5) * noise_level;
                        tensor.colors[[row, col, layer, 2]] +=
                            (rng.gen::<f32>() - 0.5) * noise_level;
                    }
                }
            }

            // Clamp to valid range
            tensor = tensor.clamp(0.0, 1.0);

            dataset.push(ColorPattern {
                tensor,
                label: class,
                description: format!("{} pattern #{}", class_names[class], sample_idx),
            });
        }
    }

    dataset
}

/// Splits a dataset into training and validation sets.
///
/// # Arguments
///
/// * `dataset` - Full dataset
/// * `train_fraction` - Fraction to use for training (e.g., 0.8)
///
/// # Returns
///
/// Tuple of (training set, validation set)
pub fn split_dataset(
    dataset: Vec<ColorPattern>,
    train_fraction: f32,
) -> (Vec<ColorPattern>, Vec<ColorPattern>) {
    let train_size = (dataset.len() as f32 * train_fraction) as usize;
    let (train, val) = dataset.split_at(train_size);
    (train.to_vec(), val.to_vec())
}

/// Shuffles a dataset in place.
///
/// # Arguments
///
/// * `dataset` - Dataset to shuffle
/// * `seed` - Random seed
pub fn shuffle_dataset(dataset: &mut [ColorPattern], seed: u64) {
    use rand::seq::SliceRandom;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    dataset.shuffle(&mut rng);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_dataset() {
        let dataset = generate_primary_color_dataset(10, 8, 8, 2, 42);

        assert_eq!(dataset.len(), 30); // 10 per class × 3 classes

        // Check that classes are present
        let mut class_counts = vec![0, 0, 0];
        for pattern in &dataset {
            class_counts[pattern.label] += 1;
        }

        assert_eq!(class_counts, vec![10, 10, 10]);
    }

    #[test]
    fn test_class_separation() {
        let dataset = generate_primary_color_dataset(5, 8, 8, 2, 42);

        // Red class should have higher red values
        for pattern in dataset.iter().filter(|p| p.label == 0) {
            let stats = pattern.tensor.statistics();
            assert!(stats.mean_rgb[0] > stats.mean_rgb[1]); // R > G
            assert!(stats.mean_rgb[0] > stats.mean_rgb[2]); // R > B
        }

        // Green class should have higher green values
        for pattern in dataset.iter().filter(|p| p.label == 1) {
            let stats = pattern.tensor.statistics();
            assert!(stats.mean_rgb[1] > stats.mean_rgb[0]); // G > R
            assert!(stats.mean_rgb[1] > stats.mean_rgb[2]); // G > B
        }

        // Blue class should have higher blue values
        for pattern in dataset.iter().filter(|p| p.label == 2) {
            let stats = pattern.tensor.statistics();
            assert!(stats.mean_rgb[2] > stats.mean_rgb[0]); // B > R
            assert!(stats.mean_rgb[2] > stats.mean_rgb[1]); // B > G
        }
    }

    #[test]
    fn test_split_dataset() {
        let dataset = generate_primary_color_dataset(10, 4, 4, 2, 42);
        let (train, val) = split_dataset(dataset, 0.8);

        assert_eq!(train.len(), 24); // 0.8 * 30
        assert_eq!(val.len(), 6); // 0.2 * 30
    }
}
