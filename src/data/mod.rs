//! Dataset and pattern generation for chromatic neural networks.

pub mod color_dataset;
pub mod pattern;

pub use color_dataset::{ColorClass, ColorDataset, ColorSample, DatasetConfig};
pub use pattern::{generate_primary_color_dataset, shuffle_dataset, split_dataset, ColorPattern};
