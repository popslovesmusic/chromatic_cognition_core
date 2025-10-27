use std::fmt::{self, Display};
use std::ops::{Add, Sub};

use ndarray::{Array3, Array4, Axis};
use rayon::prelude::*;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct ChromaticTensor {
    pub colors: Array4<f32>,
    pub certainty: Array3<f32>,
}

impl ChromaticTensor {
    pub fn new(rows: usize, cols: usize, layers: usize) -> Self {
        Self {
            colors: Array4::zeros((rows, cols, layers, 3)),
            certainty: Array3::zeros((rows, cols, layers)),
        }
    }

    pub fn from_arrays(colors: Array4<f32>, certainty: Array3<f32>) -> Self {
        assert_eq!(colors.dim().0, certainty.dim().0);
        assert_eq!(colors.dim().1, certainty.dim().1);
        assert_eq!(colors.dim().2, certainty.dim().2);
        Self { colors, certainty }
    }

    pub fn from_seed(seed: u64, rows: usize, cols: usize, layers: usize) -> Self {
        let mut tensor = Self::new(rows, cols, layers);
        let state = if seed == 0 { 1 } else { seed };
        let total = rows * cols * layers;

        tensor
            .colors
            .as_slice_mut()
            .expect("ndarray uses contiguous layout")
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, value)| {
                let step = idx as u64 + state;
                let next = lcg(step);
                *value = normalized(next);
            });

        tensor
            .certainty
            .as_slice_mut()
            .expect("ndarray uses contiguous layout")
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, value)| {
                let step = idx as u64 + state.wrapping_add(total as u64);
                let next = lcg(step);
                *value = normalized(next.wrapping_mul(3)).mul_add(0.9, 0.1);
            });

        tensor
    }

    pub fn shape(&self) -> (usize, usize, usize, usize) {
        self.colors.dim()
    }

    pub fn normalize(&self) -> Self {
        let mut colors = self.colors.clone();
        colors
            .as_slice_mut()
            .expect("contiguous")
            .par_iter_mut()
            .for_each(|value| {
                if !value.is_finite() {
                    *value = 0.0;
                }
            });
        let max_value = colors
            .as_slice()
            .expect("contiguous")
            .par_iter()
            .cloned()
            .reduce(|| 0.0, f32::max);
        if max_value > 1.0 {
            colors
                .as_slice_mut()
                .expect("contiguous")
                .par_iter_mut()
                .for_each(|value| *value /= max_value);
        }
        Self::from_arrays(colors, self.certainty.clone())
    }

    pub fn clamp(&self, min: f32, max: f32) -> Self {
        let mut colors = self.colors.clone();
        colors
            .as_slice_mut()
            .expect("contiguous")
            .par_iter_mut()
            .for_each(|value| *value = value.clamp(min, max));
        Self::from_arrays(colors, self.certainty.clone())
    }

    pub fn complement(&self) -> Self {
        let mut colors = self.colors.clone();
        colors
            .axis_iter_mut(Axis(3))
            .par_bridge()
            .for_each(|mut pixel| {
                let g = pixel[1];
                let b = pixel[2];
                pixel[1] = 1.0 - g;
                pixel[2] = 1.0 - b;
            });
        Self::from_arrays(colors, self.certainty.clone())
    }

    pub fn saturate(&self, alpha: f32) -> Self {
        let mut colors = self.colors.clone();
        colors
            .axis_iter_mut(Axis(3))
            .par_bridge()
            .for_each(|mut pixel| {
                let mean = (pixel[0] + pixel[1] + pixel[2]) / 3.0;
                for value in pixel.iter_mut() {
                    *value = mean + (*value - mean) * alpha;
                }
            });
        Self::from_arrays(colors, self.certainty.clone())
    }

    pub fn statistics(&self) -> TensorStatistics {
        let (rows, cols, layers, _) = self.colors.dim();
        let cells = (rows * cols * layers) as f32;

        let mut mean_rgb = [0.0f32; 3];
        for channel in 0..3 {
            mean_rgb[channel] = self
                .colors
                .index_axis(Axis(3), channel)
                .as_slice()
                .expect("contiguous")
                .par_iter()
                .cloned()
                .sum::<f32>()
                / cells;
        }

        let mut variance_sum = 0.0f32;
        for channel in 0..3 {
            let mean = mean_rgb[channel];
            let channel_variance = self
                .colors
                .index_axis(Axis(3), channel)
                .as_slice()
                .expect("contiguous")
                .par_iter()
                .map(|value| {
                    let diff = *value - mean;
                    diff * diff
                })
                .sum::<f32>();
            variance_sum += channel_variance;
        }

        let variance = variance_sum / (cells * 3.0);
        let mean_certainty = self
            .certainty
            .as_slice()
            .expect("contiguous")
            .par_iter()
            .cloned()
            .sum::<f32>()
            / cells;

        TensorStatistics {
            mean_rgb,
            variance,
            mean_certainty,
        }
    }
}

impl Display for ChromaticTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.statistics();
        write!(
            f,
            "ChromaticTensor {}x{}x{} mean_rgb=({:.3},{:.3},{:.3}) variance={:.5}",
            self.colors.dim().0,
            self.colors.dim().1,
            self.colors.dim().2,
            stats.mean_rgb[0],
            stats.mean_rgb[1],
            stats.mean_rgb[2],
            stats.variance,
        )
    }
}

impl Add for ChromaticTensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.colors.dim(),
            rhs.colors.dim(),
            "tensor shapes must match"
        );
        let colors = &self.colors + &rhs.colors;
        let certainty = (&self.certainty + &rhs.certainty) * 0.5;
        Self::from_arrays(colors, certainty)
    }
}

impl Sub for ChromaticTensor {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.colors.dim(),
            rhs.colors.dim(),
            "tensor shapes must match"
        );
        let colors = &self.colors - &rhs.colors;
        let certainty = (&self.certainty + &rhs.certainty) * 0.5;
        Self::from_arrays(colors, certainty)
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct TensorStatistics {
    pub mean_rgb: [f32; 3],
    pub variance: f32,
    pub mean_certainty: f32,
}

fn lcg(seed: u64) -> u64 {
    seed.wrapping_mul(1664525).wrapping_add(1013904223)
}

fn normalized(value: u64) -> f32 {
    let fraction = (value & 0xFFFF_FFFF) as f32 / (u32::MAX as f32);
    fraction.clamp(0.0, 1.0)
}
