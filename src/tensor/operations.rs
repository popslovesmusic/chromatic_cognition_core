//! Primitive chromatic tensor operations.
//!
//! This module provides the core color-space operations for chromatic tensors:
//! - [`mix`] - Additive coherence
//! - [`filter`] - Subtractive distinction
//! - [`complement`] - Hue rotation
//! - [`saturate`] - Chroma adjustment
//!
//! All operations are parallelized using rayon and log their results.

use ndarray::{Array3, Array4, Zip};

use super::{ChromaticTensor, TensorStatistics};
use crate::logging;

/// Combines two tensors through additive coherence.
///
/// Performs element-wise addition of color values and normalizes the result
/// to ensure all values remain in the valid [0.0, 1.0] range. Certainty values
/// are averaged between the two input tensors.
///
/// # Arguments
///
/// * `a` - First chromatic tensor
/// * `b` - Second chromatic tensor (must have same dimensions as `a`)
///
/// # Panics
///
/// Panics if the tensors have different dimensions.
///
/// # Examples
///
/// ```
/// use chromatic_cognition_core::{ChromaticTensor, mix};
///
/// let a = ChromaticTensor::from_seed(42, 32, 32, 4);
/// let b = ChromaticTensor::from_seed(100, 32, 32, 4);
/// let result = mix(&a, &b);
/// ```
pub fn mix(a: &ChromaticTensor, b: &ChromaticTensor) -> ChromaticTensor {
    ensure_same_shape(a, b);
    let mut colors = Array4::zeros(a.colors.dim());
    Zip::from(&mut colors)
        .and(&a.colors)
        .and(&b.colors)
        .par_for_each(|out, &lhs, &rhs| {
            *out = lhs + rhs;
        });

    let mut certainty = Array3::zeros(a.certainty.dim());
    Zip::from(&mut certainty)
        .and(&a.certainty)
        .and(&b.certainty)
        .par_for_each(|out, &lhs, &rhs| {
            *out = (lhs + rhs) * 0.5;
        });

    let tensor = ChromaticTensor::from_arrays(colors, certainty).normalize();
    log_operation("mix", &tensor.statistics());
    tensor
}

/// Applies subtractive filtering between two tensors.
///
/// Performs element-wise subtraction (a - b) and clamps the result to [0.0, 1.0].
/// This operation emphasizes distinctions between the tensors. Certainty values
/// are averaged.
///
/// # Arguments
///
/// * `a` - Base chromatic tensor
/// * `b` - Filter chromatic tensor (must have same dimensions as `a`)
///
/// # Panics
///
/// Panics if the tensors have different dimensions.
///
/// # Examples
///
/// ```
/// use chromatic_cognition_core::{ChromaticTensor, filter};
///
/// let a = ChromaticTensor::from_seed(42, 32, 32, 4);
/// let b = ChromaticTensor::from_seed(100, 32, 32, 4);
/// let result = filter(&a, &b);
/// ```
pub fn filter(a: &ChromaticTensor, b: &ChromaticTensor) -> ChromaticTensor {
    ensure_same_shape(a, b);
    let mut colors = Array4::zeros(a.colors.dim());
    Zip::from(&mut colors)
        .and(&a.colors)
        .and(&b.colors)
        .par_for_each(|out, &lhs, &rhs| {
            *out = (lhs - rhs).clamp(0.0, 1.0);
        });

    let mut certainty = Array3::zeros(a.certainty.dim());
    Zip::from(&mut certainty)
        .and(&a.certainty)
        .and(&b.certainty)
        .par_for_each(|out, &lhs, &rhs| {
            *out = (lhs + rhs) * 0.5;
        });

    let tensor = ChromaticTensor::from_arrays(colors, certainty);
    log_operation("filter", &tensor.statistics());
    tensor
}

/// Computes the chromatic complement by rotating hue 180 degrees.
///
/// Inverts the green and blue channels while keeping the red channel unchanged.
/// This creates a complementary color effect in the (g,b) plane.
///
/// # Arguments
///
/// * `a` - Chromatic tensor to complement
///
/// # Examples
///
/// ```
/// use chromatic_cognition_core::{ChromaticTensor, complement};
///
/// let tensor = ChromaticTensor::from_seed(42, 32, 32, 4);
/// let result = complement(&tensor);
/// ```
pub fn complement(a: &ChromaticTensor) -> ChromaticTensor {
    let tensor = a.complement();
    log_operation("complement", &tensor.statistics());
    tensor
}

/// Adjusts color saturation by scaling distance from the mean.
///
/// Multiplies the deviation of each color channel from the mean by `alpha`.
/// Values are clamped to [0.0, 1.0] after saturation.
///
/// # Arguments
///
/// * `a` - Chromatic tensor to saturate
/// * `alpha` - Saturation multiplier (>1.0 increases saturation, <1.0 decreases)
///
/// # Examples
///
/// ```
/// use chromatic_cognition_core::{ChromaticTensor, saturate};
///
/// let tensor = ChromaticTensor::from_seed(42, 32, 32, 4);
/// let more_saturated = saturate(&tensor, 1.5);
/// let desaturated = saturate(&tensor, 0.5);
/// ```
pub fn saturate(a: &ChromaticTensor, alpha: f32) -> ChromaticTensor {
    let tensor = a.saturate(alpha).clamp(0.0, 1.0);
    log_operation("saturate", &tensor.statistics());
    tensor
}

fn log_operation(name: &str, stats: &TensorStatistics) {
    if let Err(err) = logging::log_operation(name, stats) {
        eprintln!("failed to log tensor operation {name}: {err}");
    }
}

fn ensure_same_shape(a: &ChromaticTensor, b: &ChromaticTensor) {
    assert_eq!(a.colors.dim(), b.colors.dim(), "tensor shapes must match");
    assert_eq!(
        a.certainty.dim(),
        b.certainty.dim(),
        "tensor certainty shapes must match"
    );
}
