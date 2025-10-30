struct ChromaticSpiralUniform {
    position_x: f32,
    position_y: f32,
    coherence_score: f32,
    padding: f32,
};

@group(0) @binding(0)
var<uniform> u_chromatic: ChromaticSpiralUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );

    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

fn polar_angle(delta: vec2<f32>) -> f32 {
    return atan2(delta.y, delta.x);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center = vec2<f32>(u_chromatic.position_x, u_chromatic.position_y);
    let coherence = clamp(u_chromatic.coherence_score, 0.0, 1.0);
    let delta = in.uv - center;
    let radius = length(delta);
    let angle = polar_angle(delta);

    // Spiral modulation accentuated by the coherence score.
    let spiral = sin(12.0 * radius + angle);
    let intensity = mix(0.1, 1.0, coherence);
    let chroma = 0.5 + 0.5 * spiral * intensity;

    let red = clamp(chroma, 0.0, 1.0);
    let green = clamp(1.0 - abs(chroma - 0.5) * 2.0, 0.0, 1.0);
    let blue = clamp(1.0 - chroma, 0.0, 1.0);

    return vec4<f32>(vec3<f32>(red, green, blue), 1.0);
}
