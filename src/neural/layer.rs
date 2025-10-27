//! Neural network layer abstractions for chromatic tensors.

use crate::neural::gradient::{backward_complement, backward_filter, backward_mix, backward_saturate};
use crate::tensor::ChromaticTensor;

/// Type of chromatic operation performed by a layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChromaticOp {
    /// Additive mix operation
    Mix,
    /// Subtractive filter operation
    Filter,
    /// Hue complement operation
    Complement,
    /// Saturation adjustment operation
    Saturate,
}

/// A chromatic neural network layer.
///
/// Performs learned transformations on chromatic tensors using color-space operations.
///
/// # Architecture
///
/// ```text
/// input → mix(input, weights) → operation → add(bias) → output
/// ```
pub struct ChromaticLayer {
    /// Learnable weight tensor
    pub weights: ChromaticTensor,
    /// Learnable bias tensor
    pub bias: ChromaticTensor,
    /// Operation to apply
    pub operation: ChromaticOp,
    /// Operation parameter (e.g., saturation alpha)
    pub param: f32,
    /// Cached input for backward pass
    cached_input: Option<ChromaticTensor>,
    /// Cached pre-operation output for backward pass
    cached_pre_op: Option<ChromaticTensor>,
}

impl ChromaticLayer {
    /// Creates a new chromatic layer with random initialization.
    ///
    /// # Arguments
    ///
    /// * `rows` - Tensor height
    /// * `cols` - Tensor width
    /// * `layers` - Tensor depth
    /// * `operation` - Type of chromatic operation
    /// * `seed` - Random seed for initialization
    ///
    /// # Examples
    ///
    /// ```
    /// use chromatic_cognition_core::neural::{ChromaticLayer, ChromaticOp};
    ///
    /// let layer = ChromaticLayer::new(32, 32, 8, ChromaticOp::Mix, 42);
    /// ```
    pub fn new(
        rows: usize,
        cols: usize,
        layers: usize,
        operation: ChromaticOp,
        seed: u64,
    ) -> Self {
        Self {
            weights: ChromaticTensor::from_seed(seed, rows, cols, layers),
            bias: ChromaticTensor::from_seed(seed.wrapping_add(1000), rows, cols, layers),
            operation,
            param: 1.0, // Default parameter
            cached_input: None,
            cached_pre_op: None,
        }
    }

    /// Sets the operation parameter (e.g., saturation alpha).
    pub fn with_param(mut self, param: f32) -> Self {
        self.param = param;
        self
    }

    /// Forward pass through the layer.
    ///
    /// Computes: operation(mix(input, weights) + bias)
    ///
    /// # Arguments
    ///
    /// * `input` - Input chromatic tensor
    ///
    /// # Returns
    ///
    /// Output chromatic tensor after transformation
    pub fn forward(&mut self, input: &ChromaticTensor) -> ChromaticTensor {
        // Cache input for backward pass
        self.cached_input = Some(input.clone());

        // Step 1: Mix input with learned weights
        let pre_op = crate::tensor::operations::mix(input, &self.weights);

        // Cache pre-operation result
        self.cached_pre_op = Some(pre_op.clone());

        // Step 2: Apply operation
        let after_op = match self.operation {
            ChromaticOp::Mix => pre_op, // Already mixed
            ChromaticOp::Filter => crate::tensor::operations::filter(&pre_op, &self.weights),
            ChromaticOp::Complement => crate::tensor::operations::complement(&pre_op),
            ChromaticOp::Saturate => crate::tensor::operations::saturate(&pre_op, self.param),
        };

        // Step 3: Add bias
        after_op + self.bias.clone()
    }

    /// Backward pass through the layer.
    ///
    /// Computes gradients with respect to inputs and parameters.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient flowing back from next layer
    ///
    /// # Returns
    ///
    /// Tuple of (gradient w.r.t. input, gradient w.r.t. weights, gradient w.r.t. bias)
    pub fn backward(
        &self,
        grad_output: &ChromaticTensor,
    ) -> (ChromaticTensor, ChromaticTensor, ChromaticTensor) {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before backward");
        let pre_op = self
            .cached_pre_op
            .as_ref()
            .expect("forward must be called before backward");

        // Gradient through bias addition (pass through)
        let grad_after_op = grad_output.clone();

        // Gradient through operation
        let grad_pre_op = match self.operation {
            ChromaticOp::Mix => grad_after_op, // No-op
            ChromaticOp::Filter => {
                let (grad_a, _grad_b) = backward_filter(&grad_after_op, pre_op, &self.weights);
                grad_a
            }
            ChromaticOp::Complement => backward_complement(&grad_after_op),
            ChromaticOp::Saturate => {
                let (grad_in, _grad_alpha) = backward_saturate(&grad_after_op, pre_op, self.param);
                grad_in
            }
        };

        // Gradient through mix(input, weights)
        let (grad_input, grad_weights) = backward_mix(&grad_pre_op, input, &self.weights, pre_op);

        // Gradient for bias is just the output gradient
        let grad_bias = grad_output.clone();

        (grad_input, grad_weights, grad_bias)
    }

    /// Returns the shape of this layer's tensors.
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        self.weights.shape()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = ChromaticLayer::new(32, 32, 4, ChromaticOp::Mix, 42);
        assert_eq!(layer.shape(), (32, 32, 4, 3));
    }

    #[test]
    fn test_forward_pass() {
        let mut layer = ChromaticLayer::new(8, 8, 2, ChromaticOp::Saturate, 42);
        layer = layer.with_param(1.5);

        let input = ChromaticTensor::from_seed(100, 8, 8, 2);
        let output = layer.forward(&input);

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_backward_pass() {
        let mut layer = ChromaticLayer::new(8, 8, 2, ChromaticOp::Mix, 42);
        let input = ChromaticTensor::from_seed(100, 8, 8, 2);

        // Forward
        let _output = layer.forward(&input);

        // Backward
        let grad_output = ChromaticTensor::from_seed(200, 8, 8, 2);
        let (grad_input, grad_weights, grad_bias) = layer.backward(&grad_output);

        // Check shapes
        assert_eq!(grad_input.shape(), input.shape());
        assert_eq!(grad_weights.shape(), layer.weights.shape());
        assert_eq!(grad_bias.shape(), layer.bias.shape());
    }
}
