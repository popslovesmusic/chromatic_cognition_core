use std::io;
use std::path::Path;

use ndarray::{s, Array3};
use plotters::prelude::*;
use rayon::prelude::*;

use super::ChromaticTensor;

#[derive(Debug, Clone)]
pub struct GradientLayer {
    pub image: Array3<f32>,
}

impl GradientLayer {
    pub fn from_tensor(tensor: &ChromaticTensor) -> Self {
        let (rows, cols, layers, _) = tensor.colors.dim();
        let mut image = Array3::zeros((rows, cols, 3));

        image
            .indexed_iter_mut()
            .par_bridge()
            .for_each(|((row, col, channel), value)| {
                let mut numerator = 0.0f32;
                let mut denominator = 0.0f32;
                for layer in 0..layers {
                    let weight = tensor.certainty[[row, col, layer]].max(0.0);
                    let color = tensor.colors[[row, col, layer, channel]];
                    numerator += color * weight;
                    denominator += weight;
                }
                *value = if denominator > 0.0 {
                    (numerator / denominator).clamp(0.0, 1.0)
                } else {
                    0.0
                };
            });

        Self { image }
    }

    pub fn to_png<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let (rows, cols, _) = self.image.dim();
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let backend = BitMapBackend::new(path, (cols as u32, rows as u32));
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&RGBColor(0, 0, 0))
            .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;

        for row in 0..rows {
            for col in 0..cols {
                let pixel = self.image.slice(s![row, col, ..]);
                let color = RGBColor(
                    float_to_byte(pixel[0]),
                    float_to_byte(pixel[1]),
                    float_to_byte(pixel[2]),
                );
                drawing_area
                    .draw_pixel((col as i32, row as i32), &color)
                    .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
            }
        }

        drawing_area
            .present()
            .map_err(|err| io::Error::new(io::ErrorKind::Other, err))
    }
}

fn float_to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}
