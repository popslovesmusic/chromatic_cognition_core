//! Loss functions for training chromatic neural networks.

use crate::tensor::ChromaticTensor;

/// Computes mean squared error loss with gradients.
///
/// # Arguments
///
/// * `predicted` - Predicted chromatic tensor
/// * `target` - Target chromatic tensor
///
/// # Returns
///
/// Tuple of (loss value, gradient w.r.t. predicted)
pub fn mse_loss_with_gradients(
    predicted: &ChromaticTensor,
    target: &ChromaticTensor,
) -> (f32, ChromaticTensor) {
    let diff = predicted.clone() - target.clone();

    // Loss = mean((predicted - target)^2)
    let squared_diff = diff.colors.mapv(|x| x * x);
    let loss = squared_diff.mean().unwrap_or(0.0);

    // Gradient = 2 * (predicted - target) / N
    let n = (predicted.shape().0 * predicted.shape().1 * predicted.shape().2 * 3) as f32;
    let gradient = diff * (2.0 / n);

    (loss, gradient)
}

/// Computes cross-entropy loss for classification.
///
/// Takes the mean color of the final tensor as logits for each class.
///
/// # Arguments
///
/// * `predicted` - Predicted chromatic tensor (final layer)
/// * `label` - Target class label
/// * `num_classes` - Total number of classes
///
/// # Returns
///
/// Tuple of (loss value, gradient w.r.t. predicted colors)
pub fn cross_entropy_loss(
    predicted: &ChromaticTensor,
    label: usize,
    num_classes: usize,
) -> (f32, ChromaticTensor) {
    // Extract logits from mean RGB values
    let stats = predicted.statistics();
    let logits = stats.mean_rgb;

    // For 3-class problem, use RGB channels as logits
    // For other cases, we'd need to extend this
    assert!(num_classes <= 3, "Currently only supports up to 3 classes");

    // Softmax
    let max_logit = logits[..num_classes].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits[..num_classes].iter().map(|&x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = logits[..num_classes].iter().map(|&x| (x - max_logit).exp() / exp_sum).collect();

    // Cross-entropy loss
    let loss = -probs[label].ln();

    // Gradient: softmax gradient
    let mut grad_logits = [0.0f32; 3];
    for i in 0..num_classes {
        if i == label {
            grad_logits[i] = probs[i] - 1.0;
        } else {
            grad_logits[i] = probs[i];
        }
    }

    // Scale gradient back to tensor
    // Gradient is uniform across all spatial locations and layers
    let (rows, cols, layers, _) = predicted.shape();
    let n = (rows * cols * layers) as f32;

    let mut gradient = predicted.clone();
    for val in gradient.colors.as_slice_mut().expect("contiguous").chunks_exact_mut(3) {
        val[0] = grad_logits[0] / n;
        val[1] = grad_logits[1] / n;
        val[2] = grad_logits[2] / n;
    }

    (loss, gradient)
}

/// Computes accuracy for classification.
///
/// # Arguments
///
/// * `predicted` - Predicted chromatic tensor
/// * `label` - True label
/// * `num_classes` - Number of classes
///
/// # Returns
///
/// 1.0 if prediction is correct, 0.0 otherwise
pub fn accuracy(predicted: &ChromaticTensor, label: usize, num_classes: usize) -> f32 {
    let stats = predicted.statistics();
    let logits = stats.mean_rgb;

    // Find argmax
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..num_classes.min(3) {
        if logits[i] > max_val {
            max_val = logits[i];
            max_idx = i;
        }
    }

    if max_idx == label {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let predicted = ChromaticTensor::from_seed(42, 4, 4, 2);
        let target = ChromaticTensor::from_seed(100, 4, 4, 2);

        let (loss, gradient) = mse_loss_with_gradients(&predicted, &target);

        assert!(loss > 0.0);
        assert_eq!(gradient.shape(), predicted.shape());
    }

    #[test]
    fn test_cross_entropy() {
        let predicted = ChromaticTensor::from_seed(42, 4, 4, 2);
        let label = 0;
        let num_classes = 3;

        let (loss, gradient) = cross_entropy_loss(&predicted, label, num_classes);

        assert!(loss > 0.0);
        assert_eq!(gradient.shape(), predicted.shape());
    }

    #[test]
    fn test_accuracy() {
        let mut tensor = ChromaticTensor::from_seed(42, 4, 4, 2);

        // Manually set colors to favor class 0 (red)
        for val in tensor.colors.as_slice_mut().expect("contiguous").chunks_exact_mut(3) {
            val[0] = 1.0; // Red
            val[1] = 0.0; // Green
            val[2] = 0.0; // Blue
        }

        let acc = accuracy(&tensor, 0, 3);
        assert_eq!(acc, 1.0);

        let acc_wrong = accuracy(&tensor, 1, 3);
        assert_eq!(acc_wrong, 0.0);
    }
}
