use ndarray::{Array3, Array4, Zip};

use super::{ChromaticTensor, TensorStatistics};
use crate::logging;

pub fn mix(a: &ChromaticTensor, b: &ChromaticTensor) -> ChromaticTensor {
    ensure_same_shape(a, b);
    let mut colors = Array4::zeros(a.colors.dim());
    Zip::from(&mut colors)
        .and(&a.colors)
        .and(&b.colors)
        .par_apply(|out, &lhs, &rhs| {
            *out = lhs + rhs;
        });

    let mut certainty = Array3::zeros(a.certainty.dim());
    Zip::from(&mut certainty)
        .and(&a.certainty)
        .and(&b.certainty)
        .par_apply(|out, &lhs, &rhs| {
            *out = (lhs + rhs) * 0.5;
        });

    let tensor = ChromaticTensor::from_arrays(colors, certainty).normalize();
    log_operation("mix", &tensor.statistics());
    tensor
}

pub fn filter(a: &ChromaticTensor, b: &ChromaticTensor) -> ChromaticTensor {
    ensure_same_shape(a, b);
    let mut colors = Array4::zeros(a.colors.dim());
    Zip::from(&mut colors)
        .and(&a.colors)
        .and(&b.colors)
        .par_apply(|out, &lhs, &rhs| {
            *out = (lhs - rhs).clamp(0.0, 1.0);
        });

    let mut certainty = Array3::zeros(a.certainty.dim());
    Zip::from(&mut certainty)
        .and(&a.certainty)
        .and(&b.certainty)
        .par_apply(|out, &lhs, &rhs| {
            *out = (lhs + rhs) * 0.5;
        });

    let tensor = ChromaticTensor::from_arrays(colors, certainty);
    log_operation("filter", &tensor.statistics());
    tensor
}

pub fn complement(a: &ChromaticTensor) -> ChromaticTensor {
    let tensor = a.complement();
    log_operation("complement", &tensor.statistics());
    tensor
}

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
