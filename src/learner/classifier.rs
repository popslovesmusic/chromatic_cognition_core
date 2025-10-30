//! Color classification model
//!
//! Implements a simple MLP (Multi-Layer Perceptron) for classifying
//! ChromaticTensor inputs into 10 color classes.

use crate::data::ColorClass;
use crate::tensor::ChromaticTensor;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Configuration for the classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    /// Input size (flattened tensor dimensions)
    pub input_size: usize,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Output size (number of classes)
    pub output_size: usize,
    /// Random seed for weight initialization
    pub seed: u64,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            input_size: 16 * 16 * 4 * 3, // 16x16x4 tensor with RGB = 3072
            hidden_size: 128,
            output_size: 10, // 10 color classes
            seed: 42,
        }
    }
}

/// Trait for color classification models
pub trait ColorClassifier {
    /// Forward pass - predict class probabilities
    fn forward(&self, tensor: &ChromaticTensor) -> Array1<f32>;

    /// Predict the most likely class
    fn predict(&self, tensor: &ChromaticTensor) -> ColorClass;

    /// Compute loss and gradients for a batch
    fn compute_loss(&self, tensors: &[ChromaticTensor], labels: &[ColorClass]) -> (f32, Gradients);

    /// Update weights using gradients
    fn update_weights(&mut self, gradients: &Gradients, learning_rate: f32);

    /// Get current model parameters (for checkpointing)
    fn get_weights(&self) -> Weights;

    /// Set model parameters (for loading checkpoints)
    fn set_weights(&mut self, weights: Weights);
}

/// Model weights (for checkpointing and transfer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weights {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

/// Gradients for backpropagation
#[derive(Debug, Clone)]
pub struct Gradients {
    pub dw1: Array2<f32>,
    pub db1: Array1<f32>,
    pub dw2: Array2<f32>,
    pub db2: Array1<f32>,
}

/// Simple MLP classifier: Input → Hidden (ReLU) → Output (Softmax)
pub struct MLPClassifier {
    config: ClassifierConfig,
    // Layer 1: input → hidden
    w1: Array2<f32>, // [hidden_size, input_size]
    b1: Array1<f32>, // [hidden_size]
    // Layer 2: hidden → output
    w2: Array2<f32>, // [output_size, hidden_size]
    b2: Array1<f32>, // [output_size]
}

impl MLPClassifier {
    /// Create a new MLP classifier with random initialization
    pub fn new(config: ClassifierConfig) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

        // Xavier initialization for weights
        let w1_scale = (2.0 / config.input_size as f32).sqrt();
        let w1 = Array2::from_shape_fn((config.hidden_size, config.input_size), |_| {
            (rng.gen::<f32>() - 0.5) * 2.0 * w1_scale
        });

        let b1 = Array1::zeros(config.hidden_size);

        let w2_scale = (2.0 / config.hidden_size as f32).sqrt();
        let w2 = Array2::from_shape_fn((config.output_size, config.hidden_size), |_| {
            (rng.gen::<f32>() - 0.5) * 2.0 * w2_scale
        });

        let b2 = Array1::zeros(config.output_size);

        Self {
            config,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Flatten a ChromaticTensor into a 1D feature vector
    fn flatten_tensor(&self, tensor: &ChromaticTensor) -> Array1<f32> {
        let (rows, cols, layers) = tensor.dims();
        let mut features = Array1::zeros(rows * cols * layers * 3);

        let mut idx = 0;
        for r in 0..rows {
            for c in 0..cols {
                for l in 0..layers {
                    let rgb = tensor.get_rgb(r, c, l);
                    features[idx] = rgb[0];
                    features[idx + 1] = rgb[1];
                    features[idx + 2] = rgb[2];
                    idx += 3;
                }
            }
        }

        features
    }

    /// ReLU activation
    fn relu(x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| v.max(0.0))
    }

    /// ReLU derivative
    fn relu_derivative(x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    /// Softmax activation
    fn softmax(x: &Array1<f32>) -> Array1<f32> {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Array1<f32> = x.mapv(|v| (v - max).exp());
        let sum: f32 = exp.sum();
        exp / sum
    }

    /// Forward pass with intermediate activations
    fn forward_with_cache(&self, input: &Array1<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        // Hidden layer: z1 = W1 * x + b1
        let z1 = self.w1.dot(input) + &self.b1;
        let h1 = Self::relu(&z1);

        // Output layer: z2 = W2 * h1 + b2
        let z2 = self.w2.dot(&h1) + &self.b2;
        let output = Self::softmax(&z2);

        (output, h1, z1)
    }
}

impl ColorClassifier for MLPClassifier {
    fn forward(&self, tensor: &ChromaticTensor) -> Array1<f32> {
        let input = self.flatten_tensor(tensor);
        let (output, _, _) = self.forward_with_cache(&input);
        output
    }

    fn predict(&self, tensor: &ChromaticTensor) -> ColorClass {
        let probs = self.forward(tensor);
        let predicted_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        ColorClass::from_index(predicted_idx).unwrap_or(ColorClass::Red)
    }

    fn compute_loss(&self, tensors: &[ChromaticTensor], labels: &[ColorClass]) -> (f32, Gradients) {
        let batch_size = tensors.len();

        // Initialize gradient accumulators
        let mut dw1 = Array2::zeros(self.w1.dim());
        let mut db1 = Array1::zeros(self.b1.dim());
        let mut dw2 = Array2::zeros(self.w2.dim());
        let mut db2 = Array1::zeros(self.b2.dim());

        let mut total_loss = 0.0;

        for (tensor, &label) in tensors.iter().zip(labels.iter()) {
            let input = self.flatten_tensor(tensor);
            let (output, h1, z1) = self.forward_with_cache(&input);

            // Cross-entropy loss
            let label_idx = label as usize;
            let loss = -output[label_idx].ln();
            total_loss += loss;

            // Backward pass
            // Output layer gradient
            let mut dz2 = output.clone();
            dz2[label_idx] -= 1.0; // derivative of softmax + cross-entropy

            // Gradients for W2 and b2
            for i in 0..self.config.output_size {
                for j in 0..self.config.hidden_size {
                    dw2[[i, j]] += dz2[i] * h1[j];
                }
                db2[i] += dz2[i];
            }

            // Hidden layer gradient
            let dh1 = self.w2.t().dot(&dz2);
            let dz1 = &dh1 * &Self::relu_derivative(&z1);

            // Gradients for W1 and b1
            for i in 0..self.config.hidden_size {
                for j in 0..self.config.input_size {
                    dw1[[i, j]] += dz1[i] * input[j];
                }
                db1[i] += dz1[i];
            }
        }

        // Average gradients over batch
        let batch_size_f32 = batch_size as f32;
        dw1 /= batch_size_f32;
        db1 /= batch_size_f32;
        dw2 /= batch_size_f32;
        db2 /= batch_size_f32;

        let avg_loss = total_loss / batch_size_f32;

        (avg_loss, Gradients { dw1, db1, dw2, db2 })
    }

    fn update_weights(&mut self, gradients: &Gradients, learning_rate: f32) {
        // Gradient descent: W = W - lr * dW
        self.w1 = &self.w1 - &(&gradients.dw1 * learning_rate);
        self.b1 = &self.b1 - &(&gradients.db1 * learning_rate);
        self.w2 = &self.w2 - &(&gradients.dw2 * learning_rate);
        self.b2 = &self.b2 - &(&gradients.db2 * learning_rate);
    }

    fn get_weights(&self) -> Weights {
        Weights {
            w1: self.w1.iter().cloned().collect(),
            b1: self.b1.iter().cloned().collect(),
            w2: self.w2.iter().cloned().collect(),
            b2: self.b2.iter().cloned().collect(),
        }
    }

    fn set_weights(&mut self, weights: Weights) {
        self.w1 = Array2::from_shape_vec(
            (self.config.hidden_size, self.config.input_size),
            weights.w1,
        )
        .expect("Invalid w1 shape");

        self.b1 = Array1::from_vec(weights.b1);

        self.w2 = Array2::from_shape_vec(
            (self.config.output_size, self.config.hidden_size),
            weights.w2,
        )
        .expect("Invalid w2 shape");

        self.b2 = Array1::from_vec(weights.b2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColorDataset, DatasetConfig};

    #[test]
    fn test_mlp_creation() {
        let config = ClassifierConfig::default();
        let classifier = MLPClassifier::new(config);

        assert_eq!(classifier.w1.dim(), (128, 3072));
        assert_eq!(classifier.b1.dim(), 128);
        assert_eq!(classifier.w2.dim(), (10, 128));
        assert_eq!(classifier.b2.dim(), 10);
    }

    #[test]
    fn test_forward_pass() {
        let config = ClassifierConfig::default();
        let classifier = MLPClassifier::new(config);

        let tensor = ChromaticTensor::from_seed(42, 16, 16, 4);
        let output = classifier.forward(&tensor);

        assert_eq!(output.len(), 10);

        // Check softmax: probabilities sum to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // All probabilities should be positive
        assert!(output.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_predict() {
        let config = ClassifierConfig::default();
        let classifier = MLPClassifier::new(config);

        let tensor = ChromaticTensor::from_seed(42, 16, 16, 4);
        let prediction = classifier.predict(&tensor);

        // Should return a valid color class
        assert!(matches!(
            prediction,
            ColorClass::Red
                | ColorClass::Green
                | ColorClass::Blue
                | ColorClass::Yellow
                | ColorClass::Cyan
                | ColorClass::Magenta
                | ColorClass::Orange
                | ColorClass::Purple
                | ColorClass::White
                | ColorClass::Black
        ));
    }

    #[test]
    fn test_compute_loss_and_gradients() {
        let config = ClassifierConfig::default();
        let classifier = MLPClassifier::new(config);

        // Create small batch
        let dataset_config = DatasetConfig {
            tensor_size: (16, 16, 4),
            samples_per_class: 1,
            ..Default::default()
        };
        let dataset = ColorDataset::generate(dataset_config);

        let tensors: Vec<_> = dataset
            .samples
            .iter()
            .take(5)
            .map(|s| s.tensor.clone())
            .collect();
        let labels: Vec<_> = dataset.samples.iter().take(5).map(|s| s.label).collect();

        let (loss, gradients) = classifier.compute_loss(&tensors, &labels);

        // Loss should be positive
        assert!(loss > 0.0);
        assert!(loss.is_finite());

        // Gradients should have correct shapes
        assert_eq!(gradients.dw1.dim(), classifier.w1.dim());
        assert_eq!(gradients.db1.dim(), classifier.b1.dim());
        assert_eq!(gradients.dw2.dim(), classifier.w2.dim());
        assert_eq!(gradients.db2.dim(), classifier.b2.dim());

        // Gradients should be finite
        assert!(gradients.dw1.iter().all(|&v| v.is_finite()));
        assert!(gradients.db1.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_weight_update() {
        let config = ClassifierConfig::default();
        let mut classifier = MLPClassifier::new(config);

        let initial_w1 = classifier.w1.clone();

        // Create dummy gradients
        let dataset_config = DatasetConfig {
            tensor_size: (16, 16, 4),
            samples_per_class: 1,
            ..Default::default()
        };
        let dataset = ColorDataset::generate(dataset_config);

        let tensors: Vec<_> = dataset
            .samples
            .iter()
            .take(2)
            .map(|s| s.tensor.clone())
            .collect();
        let labels: Vec<_> = dataset.samples.iter().take(2).map(|s| s.label).collect();

        let (_, gradients) = classifier.compute_loss(&tensors, &labels);

        // Update weights
        classifier.update_weights(&gradients, 0.01);

        // Weights should have changed
        assert!(classifier
            .w1
            .iter()
            .zip(initial_w1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6));
    }
}
