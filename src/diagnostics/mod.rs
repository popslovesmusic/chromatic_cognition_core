//! Diagnostic and visualization utilities.
//!
//! This module hosts the GPU-backed chromatic visualization pipeline that
//! renders coherence diagnostics on top of the chromatic tensor field.  It
//! exposes helper types for configuring the WebGPU (wgpu) pipeline and the
//! uniform bridge shared with the WGSL shader.

pub mod gpu;

pub use gpu::{ChromaticSpiralPipeline, ChromaticSpiralUniform};
