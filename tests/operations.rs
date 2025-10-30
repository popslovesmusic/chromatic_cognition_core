use chromatic_cognition_core::{
    complement, filter, mix, mse_loss, saturate, ChromaticTensor, GradientLayer,
};
use ndarray::{Array3, Array4};

fn sample_tensor(values: [[f32; 3]; 2]) -> ChromaticTensor {
    let colors = Array4::from_shape_vec((1, 1, 2, 3), values.into_iter().flatten().collect())
        .expect("shape matches");
    let certainty = Array3::from_elem((1, 1, 2), 1.0);
    ChromaticTensor::from_arrays(colors, certainty)
}

#[test]
fn mix_adds_and_normalizes() {
    let a = sample_tensor([[0.2, 0.4, 0.6], [0.1, 0.2, 0.3]]);
    let b = sample_tensor([[0.3, 0.3, 0.3], [0.2, 0.5, 0.7]]);
    let mixed = mix(&a, &b);
    let data = mixed.colors;
    assert!((data[[0, 0, 0, 0]] - 0.5).abs() < 1e-6);
    assert!((data[[0, 0, 0, 1]] - 0.7).abs() < 1e-6);
    assert!((data[[0, 0, 1, 2]] - 1.0).abs() < 1e-6);
}

#[test]
fn filter_subtracts_and_clamps() {
    let a = sample_tensor([[0.4, 0.5, 0.6], [0.8, 0.1, 0.3]]);
    let b = sample_tensor([[0.2, 0.7, 0.1], [0.4, 0.5, 0.9]]);
    let filtered = filter(&a, &b);
    let data = filtered.colors;
    assert!((data[[0, 0, 0, 0]] - 0.2).abs() < 1e-6);
    assert!((data[[0, 0, 0, 1]] - 0.0).abs() < 1e-6);
    assert!((data[[0, 0, 1, 2]] - 0.0).abs() < 1e-6);
}

#[test]
fn complement_inverts_green_and_blue() {
    let tensor = sample_tensor([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]);
    let complemented = complement(&tensor);
    let data = complemented.colors;
    assert!((data[[0, 0, 0, 0]] - 0.1).abs() < 1e-6);
    assert!((data[[0, 0, 0, 1]] - 0.8).abs() < 1e-6);
    assert!((data[[0, 0, 1, 2]] - 0.3).abs() < 1e-6);
}

#[test]
fn saturate_stretches_chroma() {
    let tensor = sample_tensor([[0.3, 0.4, 0.5], [0.2, 0.4, 0.6]]);
    let saturated = saturate(&tensor, 1.5);
    let data = saturated.colors;
    assert!(data[[0, 0, 0, 0]] < tensor.colors[[0, 0, 0, 0]]);
    assert!(data[[0, 0, 0, 2]] > tensor.colors[[0, 0, 0, 2]]);
}

#[test]
fn gradient_layer_averages_across_layers() {
    let tensor = sample_tensor([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]);
    let gradient = GradientLayer::from_tensor(&tensor);
    println!(
        "R: {}, G: {}, B: {}",
        gradient.image[[0, 0, 0]],
        gradient.image[[0, 0, 1]],
        gradient.image[[0, 0, 2]]
    );
    println!("Expected: R: 0.5, G: 0.5, B: 0.5");
    // The test expects to average [0.0, 0.5, 1.0] and [1.0, 0.5, 0.0] across 2 layers
    // Result should be [(0.0+1.0)/2, (0.5+0.5)/2, (1.0+0.0)/2] = [0.5, 0.5, 0.5]
    assert!((gradient.image[[0, 0, 0]] - 0.5).abs() < 1e-6);
    assert!((gradient.image[[0, 0, 1]] - 0.5).abs() < 1e-6);
    assert!((gradient.image[[0, 0, 2]] - 0.5).abs() < 1e-6);
}

#[test]
fn mse_loss_computes_mean_squared_error() {
    let a = sample_tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
    let b = sample_tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]);
    let metrics = mse_loss(&a, &b);
    assert!(metrics.loss > 0.0);
    assert!(metrics.mean_rgb[0] > 0.0);
}
