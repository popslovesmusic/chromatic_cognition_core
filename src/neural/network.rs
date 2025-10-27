//! Chromatic neural network architecture.

use crate::neural::layer::{ChromaticLayer, ChromaticOp};
use crate::neural::loss::{accuracy, cross_entropy_loss};
use crate::neural::optimizer::SGDOptimizer;
use crate::tensor::ChromaticTensor;

/// A chromatic neural network for classification.
///
/// Stacks multiple chromatic layers to learn complex color patterns.
pub struct ChromaticNetwork {
    layers: Vec<ChromaticLayer>,
    num_classes: usize,
}

impl ChromaticNetwork {
    /// Creates a new chromatic network.
    ///
    /// # Arguments
    ///
    /// * `layers` - Vector of chromatic layers
    /// * `num_classes` - Number of output classes
    pub fn new(layers: Vec<ChromaticLayer>, num_classes: usize) -> Self {
        Self { layers, num_classes }
    }

    /// Creates a simple 2-layer network for experiments.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Size of input tensors (rows, cols, layers)
    /// * `num_classes` - Number of output classes
    /// * `seed` - Random seed
    pub fn simple(input_size: (usize, usize, usize), num_classes: usize, seed: u64) -> Self {
        let (rows, cols, layers) = input_size;

        let layer1 = ChromaticLayer::new(rows, cols, layers, ChromaticOp::Saturate, seed)
            .with_param(1.2);
        let layer2 = ChromaticLayer::new(rows, cols, layers, ChromaticOp::Mix, seed + 1);

        Self::new(vec![layer1, layer2], num_classes)
    }

    /// Forward pass through the network.
    ///
    /// # Arguments
    ///
    /// * `input` - Input chromatic tensor
    ///
    /// # Returns
    ///
    /// Output chromatic tensor
    pub fn forward(&mut self, input: &ChromaticTensor) -> ChromaticTensor {
        let mut activation = input.clone();

        for layer in &mut self.layers {
            activation = layer.forward(&activation);
        }

        activation
    }

    /// Computes loss and gradients for a single sample.
    ///
    /// # Arguments
    ///
    /// * `input` - Input chromatic tensor
    /// * `label` - Target class label
    ///
    /// # Returns
    ///
    /// Tuple of (loss, accuracy for this sample)
    pub fn compute_loss(&mut self, input: &ChromaticTensor, label: usize) -> (f32, f32) {
        // Forward pass
        let output = self.forward(input);

        // Compute loss
        let (loss, _grad) = cross_entropy_loss(&output, label, self.num_classes);

        // Compute accuracy
        let acc = accuracy(&output, label, self.num_classes);

        (loss, acc)
    }

    /// Trains the network for one step.
    ///
    /// # Arguments
    ///
    /// * `input` - Input chromatic tensor
    /// * `label` - Target class label
    /// * `optimizer` - Optimizer to use for updates
    ///
    /// # Returns
    ///
    /// Tuple of (loss, accuracy)
    pub fn train_step(
        &mut self,
        input: &ChromaticTensor,
        label: usize,
        optimizer: &mut SGDOptimizer,
    ) -> (f32, f32) {
        // Forward pass
        let output = self.forward(input);

        // Compute loss and gradient
        let (loss, grad_output) = cross_entropy_loss(&output, label, self.num_classes);

        // Backward pass - collect gradients first
        let mut layer_gradients = Vec::new();
        let mut grad = grad_output;

        for layer in self.layers.iter().rev() {
            let (grad_input, grad_weights, grad_bias) = layer.backward(&grad);
            layer_gradients.push((grad_weights, grad_bias));
            grad = grad_input;
        }

        // Now update parameters
        layer_gradients.reverse();
        for (layer_idx, (grad_weights, grad_bias)) in layer_gradients.into_iter().enumerate() {
            optimizer.step(
                &format!("layer{}_weights", layer_idx),
                &mut self.layers[layer_idx].weights,
                &grad_weights,
            );
            optimizer.step(
                &format!("layer{}_bias", layer_idx),
                &mut self.layers[layer_idx].bias,
                &grad_bias,
            );
        }

        // Compute accuracy
        let acc = accuracy(&output, label, self.num_classes);

        (loss, acc)
    }

    /// Evaluates the network on a batch of samples.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Batch of input tensors
    /// * `labels` - Batch of labels
    ///
    /// # Returns
    ///
    /// Tuple of (average loss, average accuracy)
    pub fn evaluate(&mut self, inputs: &[ChromaticTensor], labels: &[usize]) -> (f32, f32) {
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;

        for (input, &label) in inputs.iter().zip(labels.iter()) {
            let (loss, acc) = self.compute_loss(input, label);
            total_loss += loss;
            total_acc += acc;
        }

        let n = inputs.len() as f32;
        (total_loss / n, total_acc / n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let net = ChromaticNetwork::simple((16, 16, 4), 3, 42);
        assert_eq!(net.layers.len(), 2);
        assert_eq!(net.num_classes, 3);
    }

    #[test]
    fn test_forward_pass() {
        let mut net = ChromaticNetwork::simple((8, 8, 2), 3, 42);
        let input = ChromaticTensor::from_seed(100, 8, 8, 2);

        let output = net.forward(&input);
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_train_step() {
        let mut net = ChromaticNetwork::simple((8, 8, 2), 3, 42);
        let mut optimizer = SGDOptimizer::new(0.01, 0.0, 0.0);

        let input = ChromaticTensor::from_seed(100, 8, 8, 2);
        let label = 0;

        let (loss1, _acc1) = net.train_step(&input, label, &mut optimizer);
        let (loss2, _acc2) = net.train_step(&input, label, &mut optimizer);

        // Loss should decrease (or at least change) after training
        assert_ne!(loss1, loss2);
    }
}
