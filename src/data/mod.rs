//! Dataset and pattern generation for chromatic neural networks.

pub mod pattern;

pub use pattern::{ColorPattern, generate_primary_color_dataset, shuffle_dataset, split_dataset};
