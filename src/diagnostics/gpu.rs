//! GPU-backed visualization primitives for diagnostics.
//!
//! The types in this module prepare the [`wgpu`](https://crates.io/crates/wgpu)
//! pipeline required to render the chromatic coherence spiral.  The
//! [`ChromaticSpiralUniform`] structure acts as the CPU to GPU bridge that feeds
//! the WGSL shader with the spatial coordinates and coherence score computed by
//! the diagnostics engine.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Uniform payload shared with the WGSL shader.
///
/// The layout uses `#[repr(C)]` to ensure the uniform buffer matches the
/// WGSL-side expectation and includes an explicit padding slot so that the
/// total size stays aligned to 16 bytes as required by WebGPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ChromaticSpiralUniform {
    /// Normalized X coordinate within the diagnostics viewport.
    pub position_x: f32,
    /// Normalized Y coordinate within the diagnostics viewport.
    pub position_y: f32,
    /// Aggregate coherence score emitted by the CPU diagnostics pipeline.
    pub coherence_score: f32,
    /// Padding field to maintain 16-byte alignment for the uniform block.
    pub _padding: f32,
}

impl ChromaticSpiralUniform {
    /// Creates a new uniform payload using the provided coordinates and
    /// coherence score.
    pub fn new(position_x: f32, position_y: f32, coherence_score: f32) -> Self {
        Self {
            position_x,
            position_y,
            coherence_score,
            _padding: 0.0,
        }
    }
}

impl Default for ChromaticSpiralUniform {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

/// GPU resources for the chromatic spiral visualization.
///
/// The pipeline owns the WGSL shader module alongside the uniform buffer and
/// bind group that will be shared with the render pass.  Renderer
/// implementations can retrieve references to each resource and compose them
/// with their swap chain or render target of choice.
#[allow(dead_code)]
pub struct ChromaticSpiralPipeline {
    shader: wgpu::ShaderModule,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    uniform_bind_group: wgpu::BindGroup,
}

impl ChromaticSpiralPipeline {
    /// Loads the WGSL shader module and prepares the uniform bridge resources.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chromatic-spiral-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("chromatic_spiral.wgsl"))),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chromatic-spiral-uniform-buffer"),
            contents: bytemuck::bytes_of(&ChromaticSpiralUniform::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("chromatic-spiral-uniform-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chromatic-spiral-uniform-bind-group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            shader,
            uniform_buffer,
            uniform_bind_group_layout,
            uniform_bind_group,
        }
    }

    /// Writes the latest diagnostics payload into the GPU uniform buffer.
    pub fn update_uniform(&self, queue: &wgpu::Queue, uniform: &ChromaticSpiralUniform) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniform));
    }

    /// Returns the shader module backing the chromatic spiral pipeline.
    pub fn shader(&self) -> &wgpu::ShaderModule {
        &self.shader
    }

    /// Returns the uniform buffer used to store the diagnostics payload.
    pub fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform_buffer
    }

    /// Returns the bind group layout for the chromatic spiral uniform.
    pub fn uniform_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.uniform_bind_group_layout
    }

    /// Returns the bind group that binds the chromatic spiral uniform buffer.
    pub fn uniform_bind_group(&self) -> &wgpu::BindGroup {
        &self.uniform_bind_group
    }
}
