//! Optimizers for training chromatic neural networks.

use std::collections::HashMap;

use crate::tensor::ChromaticTensor;

/// Stochastic Gradient Descent optimizer with momentum.
///
/// Implements the update rule:
/// ```text
/// velocity = momentum * velocity + learning_rate * gradient
/// parameter = parameter - velocity
/// ```
pub struct SGDOptimizer {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum coefficient (0.0 = no momentum, 0.9 = strong momentum)
    pub momentum: f32,
    /// Weight decay for L2 regularization
    pub weight_decay: f32,
    /// Velocity terms for each parameter (momentum accumulation)
    velocities: HashMap<String, ChromaticTensor>,
}

impl SGDOptimizer {
    /// Creates a new SGD optimizer.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Step size for parameter updates
    /// * `momentum` - Momentum coefficient (typically 0.9)
    /// * `weight_decay` - L2 regularization strength (typically 0.0001)
    ///
    /// # Examples
    ///
    /// ```
    /// use chromatic_cognition_core::neural::SGDOptimizer;
    ///
    /// let optimizer = SGDOptimizer::new(0.01, 0.9, 0.0001);
    /// ```
    pub fn new(learning_rate: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            velocities: HashMap::new(),
        }
    }

    /// Updates a parameter using accumulated gradients.
    ///
    /// # Arguments
    ///
    /// * `param_name` - Unique identifier for this parameter
    /// * `param` - Parameter tensor to update (modified in-place)
    /// * `gradient` - Gradient tensor
    pub fn step(&mut self, param_name: &str, param: &mut ChromaticTensor, gradient: &ChromaticTensor) {
        // Get or initialize velocity for this parameter
        let velocity = self
            .velocities
            .entry(param_name.to_string())
            .or_insert_with(|| {
                ChromaticTensor::new(
                    param.shape().0,
                    param.shape().1,
                    param.shape().2,
                )
            });

        // Apply weight decay (L2 regularization)
        let mut grad_with_decay = gradient.clone();
        if self.weight_decay > 0.0 {
            grad_with_decay = grad_with_decay + param.clone() * self.weight_decay;
        }

        // Update velocity: v = momentum * v + lr * gradient
        *velocity = velocity.clone() * self.momentum + grad_with_decay * self.learning_rate;

        // Update parameter: param = param - velocity
        *param = param.clone() - velocity.clone();
    }

    /// Resets all accumulated velocities.
    pub fn zero_grad(&mut self) {
        self.velocities.clear();
    }
}

/// Adam optimizer (Adaptive Moment Estimation).
///
/// Implements adaptive learning rates for each parameter.
pub struct AdamOptimizer {
    /// Learning rate
    pub learning_rate: f32,
    /// Exponential decay rate for first moment (typically 0.9)
    pub beta1: f32,
    /// Exponential decay rate for second moment (typically 0.999)
    pub beta2: f32,
    /// Small constant for numerical stability
    pub epsilon: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// First moment estimates (mean of gradients)
    first_moments: HashMap<String, ChromaticTensor>,
    /// Second moment estimates (variance of gradients)
    second_moments: HashMap<String, ChromaticTensor>,
    /// Time step counter
    t: usize,
}

impl AdamOptimizer {
    /// Creates a new Adam optimizer.
    pub fn new(learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            t: 0,
        }
    }

    /// Updates a parameter using Adam algorithm.
    pub fn step(&mut self, param_name: &str, param: &mut ChromaticTensor, gradient: &ChromaticTensor) {
        self.t += 1;

        // Get or initialize moments
        let m = self
            .first_moments
            .entry(param_name.to_string())
            .or_insert_with(|| {
                ChromaticTensor::new(param.shape().0, param.shape().1, param.shape().2)
            });

        let v = self
            .second_moments
            .entry(param_name.to_string())
            .or_insert_with(|| {
                ChromaticTensor::new(param.shape().0, param.shape().1, param.shape().2)
            });

        // Apply weight decay
        let mut grad_with_decay = gradient.clone();
        if self.weight_decay > 0.0 {
            grad_with_decay = grad_with_decay + param.clone() * self.weight_decay;
        }

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        *m = m.clone() * self.beta1 + grad_with_decay.clone() * (1.0 - self.beta1);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        let grad_squared = element_wise_square(&grad_with_decay);
        *v = v.clone() * self.beta2 + grad_squared * (1.0 - self.beta2);

        // Bias correction
        let m_hat = m.clone() * (1.0 / (1.0 - self.beta1.powi(self.t as i32)));
        let v_hat = v.clone() * (1.0 / (1.0 - self.beta2.powi(self.t as i32)));

        // Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
        let v_sqrt = element_wise_sqrt(&v_hat);
        let denominator = v_sqrt + ChromaticTensor::new(param.shape().0, param.shape().1, param.shape().2) * self.epsilon;
        let update = element_wise_divide(&m_hat, &denominator) * self.learning_rate;

        *param = param.clone() - update;
    }

    /// Resets all accumulated moments.
    pub fn zero_grad(&mut self) {
        self.first_moments.clear();
        self.second_moments.clear();
        self.t = 0;
    }
}

// Helper functions for element-wise operations
fn element_wise_square(tensor: &ChromaticTensor) -> ChromaticTensor {
    let mut result = tensor.clone();
    for val in result.colors.as_slice_mut().expect("contiguous") {
        *val = *val * *val;
    }
    result
}

fn element_wise_sqrt(tensor: &ChromaticTensor) -> ChromaticTensor {
    let mut result = tensor.clone();
    for val in result.colors.as_slice_mut().expect("contiguous") {
        *val = val.sqrt();
    }
    result
}

fn element_wise_divide(a: &ChromaticTensor, b: &ChromaticTensor) -> ChromaticTensor {
    let mut result = a.clone();
    let a_slice = result.colors.as_slice_mut().expect("contiguous");
    let b_slice = b.colors.as_slice().expect("contiguous");

    for (a_val, &b_val) in a_slice.iter_mut().zip(b_slice.iter()) {
        *a_val = *a_val / b_val.max(1e-10); // Avoid division by zero
    }
    result
}

// Implement scalar multiplication for ChromaticTensor
impl std::ops::Mul<f32> for ChromaticTensor {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        let mut result = self;
        for val in result.colors.as_slice_mut().expect("contiguous") {
            *val *= scalar;
        }
        for val in result.certainty.as_slice_mut().expect("contiguous") {
            *val *= scalar;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_optimizer() {
        let mut optimizer = SGDOptimizer::new(0.01, 0.0, 0.0);
        let mut param = ChromaticTensor::from_seed(42, 4, 4, 2);
        let gradient = ChromaticTensor::from_seed(100, 4, 4, 2);

        let initial_stats = param.statistics();

        // Perform update
        optimizer.step("weight", &mut param, &gradient);

        let updated_stats = param.statistics();

        // Parameters should have changed
        assert_ne!(initial_stats.mean_rgb, updated_stats.mean_rgb);
    }
}
