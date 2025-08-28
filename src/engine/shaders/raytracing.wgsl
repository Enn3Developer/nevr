struct Camera {
    view_proj: mat4x4<f32>,
    view_inverse: mat4x4<f32>,
    proj_inverse: mat4x4<f32>,
    aperture: f32,
    focus_distance: f32,
    samples: u32,
    bounces: u32,
}

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<storage> objects: array<u32>;
@group(0) @binding(2) var<storage> indices: array<u32>;
@group(0) @binding(3) var<storage> vertices: array<vec4<f32>>;
@group(0) @binding(4) var<storage> normals: array<vec4<f32>>;

@group(1) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(1) var view_output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
//    if any(global_id.xy >= vec2u(view.viewport.zw)) {
//        return;
//    }

    textureStore(view_output, global_id.xy, vec4(1.0));
}