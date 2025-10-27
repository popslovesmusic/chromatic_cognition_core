//! Gradient computation for chromatic operations.
//!
//! This module implements backpropagation for all chromatic tensor operations,
//! enabling gradient-based optimization of chromatic neural networks.

use ndarray::{Array4, Zip};
use rayon::prelude::*;

use crate::tensor::ChromaticTensor;

/// Computes gradients for the mix operation.
///
/// Forward: out = normalize(a + b)
/// Backward: ∂L/∂a = ∂L/∂out * scale_factor
///          ∂L/∂b = ∂L/∂out * scale_factor
///
/// # Arguments
///
/// * `grad_output` - Gradient flowing back from the output
/// * `input_a` - First input tensor (for shape reference)
/// * `input_b` - Second input tensor (for shape reference)
/// * `output` - The output tensor from forward pass
///
/// # Returns
///
/// Tuple of (gradient w.r.t. a, gradient w.r.t. b)
pub fn backward_mix(
    grad_output: &ChromaticTensor,
    _input_a: &ChromaticTensor,
    _input_b: &ChromaticTensor,
    output: &ChromaticTensor,
) -> (ChromaticTensor, ChromaticTensor) {
    let (_rows, _cols, _layers, _) = grad_output.shape();

    // For normalization, we need the max value that was used in forward pass
    // Approximation: assume gradient is scaled by the normalization factor
    let max_value = output
        .colors
        .as_slice()
        .expect("contiguous")
        .par_iter()
        .cloned()
        .reduce(|| 0.0, f32::max)
        .max(1.0);

    let scale = 1.0 / max_value;

    // Gradient flows equally to both inputs, scaled by normalization
    let mut grad_a_colors = grad_output.colors.clone();
    let mut grad_b_colors = grad_output.colors.clone();

    grad_a_colors
        .as_slice_mut()
        .expect("contiguous")
        .par_iter_mut()
        .for_each(|v| *v *= scale);

    grad_b_colors
        .as_slice_mut()
        .expect("contiguous")
        .par_iter_mut()
        .for_each(|v| *v *= scale);

    // Certainty gradients (averaged in forward pass)
    let grad_certainty = grad_output.certainty.mapv(|v| v * 0.5);

    (
        ChromaticTensor::from_arrays(grad_a_colors, grad_certainty.clone()),
        ChromaticTensor::from_arrays(grad_b_colors, grad_certainty),
    )
}

/// Computes gradients for the filter operation.
///
/// Forward: out = clamp(a - b, 0, 1)
/// Backward: ∂L/∂a = ∂L/∂out * (mask where a > b)
///          ∂L/∂b = -∂L/∂out * (mask where a > b)
///
/// # Arguments
///
/// * `grad_output` - Gradient flowing back from the output
/// * `input_a` - First input tensor
/// * `input_b` - Second input tensor
///
/// # Returns
///
/// Tuple of (gradient w.r.t. a, gradient w.r.t. b)
pub fn backward_filter(
    grad_output: &ChromaticTensor,
    input_a: &ChromaticTensor,
    input_b: &ChromaticTensor,
) -> (ChromaticTensor, ChromaticTensor) {
    let mut grad_a_colors = Array4::zeros(grad_output.colors.dim());
    let mut grad_b_colors = Array4::zeros(grad_output.colors.dim());

    // Compute mask: where was (a - b) > 0?
    Zip::from(&mut grad_a_colors)
        .and(&grad_output.colors)
        .and(&input_a.colors)
        .and(&input_b.colors)
        .par_for_each(|grad_a, &grad_out, &a, &b| {
            if a > b {
                *grad_a = grad_out;
            }
        });

    Zip::from(&mut grad_b_colors)
        .and(&grad_output.colors)
        .and(&input_a.colors)
        .and(&input_b.colors)
        .par_for_each(|grad_b, &grad_out, &a, &b| {
            if a > b {
                *grad_b = -grad_out;
            }
        });

    // Certainty gradients (averaged in forward pass)
    let grad_certainty = grad_output.certainty.mapv(|v| v * 0.5);

    (
        ChromaticTensor::from_arrays(grad_a_colors, grad_certainty.clone()),
        ChromaticTensor::from_arrays(grad_b_colors, grad_certainty),
    )
}

/// Computes gradients for the complement operation.
///
/// Forward: out[r] = in[r], out[g] = 1 - in[g], out[b] = 1 - in[b]
/// Backward: ∂L/∂in[r] = ∂L/∂out[r]
///          ∂L/∂in[g] = -∂L/∂out[g]
///          ∂L/∂in[b] = -∂L/∂out[b]
///
/// # Arguments
///
/// * `grad_output` - Gradient flowing back from the output
///
/// # Returns
///
/// Gradient w.r.t. input
pub fn backward_complement(grad_output: &ChromaticTensor) -> ChromaticTensor {
    let (rows, cols, layers, _) = grad_output.shape();
    let mut grad_input_colors = grad_output.colors.clone();

    // Flip sign for green and blue channels
    for row in 0..rows {
        for col in 0..cols {
            for layer in 0..layers {
                grad_input_colors[[row, col, layer, 1]] *= -1.0; // Green
                grad_input_colors[[row, col, layer, 2]] *= -1.0; // Blue
            }
        }
    }

    ChromaticTensor::from_arrays(grad_input_colors, grad_output.certainty.clone())
}

/// Computes gradients for the saturate operation.
///
/// Forward: out = mean + (in - mean) * alpha
/// Backward: ∂L/∂in = ∂L/∂out * alpha
///          ∂L/∂alpha = ∂L/∂out * (in - mean)
///
/// # Arguments
///
/// * `grad_output` - Gradient flowing back from the output
/// * `input` - Input tensor
/// * `alpha` - Saturation parameter
///
/// # Returns
///
/// Tuple of (gradient w.r.t. input, gradient w.r.t. alpha)
pub fn backward_saturate(
    grad_output: &ChromaticTensor,
    input: &ChromaticTensor,
    alpha: f32,
) -> (ChromaticTensor, f32) {
    let (rows, cols, layers, _) = grad_output.shape();
    let mut grad_input_colors = Array4::zeros(grad_output.colors.dim());
    let mut grad_alpha = 0.0f32;

    for row in 0..rows {
        for col in 0..cols {
            for layer in 0..layers {
                let r = input.colors[[row, col, layer, 0]];
                let g = input.colors[[row, col, layer, 1]];
                let b = input.colors[[row, col, layer, 2]];
                let mean = (r + g + b) / 3.0;

                // Gradient w.r.t. input: scaled by alpha
                grad_input_colors[[row, col, layer, 0]] =
                    grad_output.colors[[row, col, layer, 0]] * alpha;
                grad_input_colors[[row, col, layer, 1]] =
                    grad_output.colors[[row, col, layer, 1]] * alpha;
                grad_input_colors[[row, col, layer, 2]] =
                    grad_output.colors[[row, col, layer, 2]] * alpha;

                // Gradient w.r.t. alpha: sum over all (input - mean) * grad_output
                grad_alpha += grad_output.colors[[row, col, layer, 0]] * (r - mean);
                grad_alpha += grad_output.colors[[row, col, layer, 1]] * (g - mean);
                grad_alpha += grad_output.colors[[row, col, layer, 2]] * (b - mean);
            }
        }
    }

    (
        ChromaticTensor::from_arrays(grad_input_colors, grad_output.certainty.clone()),
        grad_alpha,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, Array4};

    fn sample_tensor(values: [[f32; 3]; 2]) -> ChromaticTensor {
        let colors = Array4::from_shape_vec((1, 1, 2, 3), values.into_iter().flatten().collect())
            .expect("shape matches");
        let certainty = Array3::from_elem((1, 1, 2), 1.0);
        ChromaticTensor::from_arrays(colors, certainty)
    }

    #[test]
    fn test_backward_complement() {
        let grad_out = sample_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let grad_in = backward_complement(&grad_out);

        // Red unchanged, green/blue negated
        assert_eq!(grad_in.colors[[0, 0, 0, 0]], 1.0);
        assert_eq!(grad_in.colors[[0, 0, 0, 1]], -2.0);
        assert_eq!(grad_in.colors[[0, 0, 0, 2]], -3.0);
        assert_eq!(grad_in.colors[[0, 0, 1, 0]], 4.0);
        assert_eq!(grad_in.colors[[0, 0, 1, 1]], -5.0);
        assert_eq!(grad_in.colors[[0, 0, 1, 2]], -6.0);
    }

    #[test]
    fn test_backward_saturate() {
        let input = sample_tensor([[0.3, 0.4, 0.5], [0.2, 0.4, 0.6]]);
        let grad_out = sample_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
        let alpha = 1.5;

        let (grad_in, grad_alpha) = backward_saturate(&grad_out, &input, alpha);

        // Gradients should be scaled by alpha
        assert!((grad_in.colors[[0, 0, 0, 0]] - 1.5).abs() < 1e-5);
        assert!(grad_alpha != 0.0); // Should accumulate gradient
    }
}
