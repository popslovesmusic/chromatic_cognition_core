use rayon::prelude::*;

use crate::tensor::ChromaticTensor;

#[derive(Debug, Clone, Copy)]
pub struct TrainingMetrics {
    pub loss: f32,
    pub mean_rgb: [f32; 3],
    pub variance: f32,
}

pub fn mse_loss(current: &ChromaticTensor, target: &ChromaticTensor) -> TrainingMetrics {
    assert_eq!(
        current.colors.dim(),
        target.colors.dim(),
        "shape mismatch for MSE"
    );

    let current_slice = current
        .colors
        .as_slice()
        .expect("ndarray uses contiguous layout for Array4");
    let target_slice = target
        .colors
        .as_slice()
        .expect("ndarray uses contiguous layout for Array4");

    let loss = current_slice
        .par_iter()
        .zip(target_slice.par_iter())
        .map(|(&lhs, &rhs)| {
            let diff = lhs - rhs;
            diff * diff
        })
        .sum::<f32>()
        / current_slice.len() as f32;

    let stats = current.statistics();

    TrainingMetrics {
        loss,
        mean_rgb: stats.mean_rgb,
        variance: stats.variance,
    }
}
