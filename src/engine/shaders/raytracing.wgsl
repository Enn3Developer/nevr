#import bevy_render::view::View

struct Camera {
    view_proj: mat4x4<f32>,
    view_inverse: mat4x4<f32>,
    proj_inverse: mat4x4<f32>,
    aperture: f32,
    focus_distance: f32,
    samples: u32,
    bounces: u32,
}

struct Light {
    ambient: vec4<f32>,
    direction: vec4<f32>,
    sky_color: vec4<f32>,
}

const RAY_T_MIN = 0.01f;
const RAY_T_MAX = 100000.0f;

const RAY_NO_CULL = 0xFFu;

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<storage, read> objects: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> normals: array<vec4<f32>>;

@group(1) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(1) var view_output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var<uniform> light: Light;
@group(1) @binding(3) var<uniform> view: View;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.viewport.zw)) {
        return;
    }

    var pixel_color = vec3(0.0);
    var seed = init_random_seed(init_random_seed(global_id.x, global_id.y), camera.samples * camera.bounces);
    var pixel_seed = init_random_seed(camera.samples * camera.bounces, camera.samples);

    for (var i = u32(0); i < camera.samples; i++) {
        var ray_color = light.sky_color.rgb;
        var jitter = vec2(0.5);
        if (i > 0) {
            jitter = vec2(random_float(&pixel_seed), random_float(&pixel_seed));
        }
        let pixel_center = vec2<f32>(global_id.xy) + jitter;
        let in_uv = pixel_center / vec2(view.viewport.zw);
        let d = in_uv * 2.0 - 1.0;

        let offset = camera.aperture / 2.0 * random_in_unit_disk(&seed);
        let origin = camera.view_inverse * vec4(offset, 0.0, 1.0);
        let camera_target = camera.proj_inverse * vec4(d.x, d.y, 1.0, 1.0);
        let direction = camera.view_inverse * vec4(normalize(camera_target.xyz * camera.focus_distance - vec3(offset, 0.0)), 0.0);

        let hit = trace_ray(origin.xyz, direction.xyz, 0.001, 10000.0, RAY_FLAG_CULL_NO_OPAQUE | RAY_FLAG_SKIP_AABBS);
        if hit.kind != RAY_QUERY_INTERSECTION_NONE {
            let barycentrics = vec3(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics.x, hit.barycentrics.y);

            let object = objects[hit.instance_custom_data];
            let index = indices[object + hit.primitive_index];
            let n0 = normals[index.x].xyz;
            let n1 = normals[index.y].xyz;
            let n2 = normals[index.z].xyz;

            let normal = n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z;
            let world_normal = normalize((hit.world_to_object * vec4(normal, 1.0)).xyz);

            ray_color = abs(n1);
        }

        pixel_color += ray_color;
    }

    pixel_color = pixel_color / f32(camera.samples);

    textureStore(view_output, global_id.xy, vec4(pixel_color, 1.0));
}

fn trace_ray(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32) -> RayIntersection {
    let ray = RayDesc(ray_flag, RAY_NO_CULL, ray_t_min, ray_t_max, ray_origin, ray_direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, ray);
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq);
}

fn init_random_seed(val0: u32, val1: u32) -> u32 {
    var v0 = val0;
    var v1 = val1;
    var s0 = u32(0);

    for (var n = 0; n < 16; n++) {
        s0 += u32(0x9e3779b9);
        v0 += ((v1 << 4) + u32(0xa341316c)) ^ (v1 + s0) ^ ((v1 >> 5) + u32(0xc8013ea4));
        v1 += ((v0 << 4) + u32(0xad90777d)) ^ (v0 + s0) ^ ((v0 >> 5) + u32(0x7e95761e));
    }

    return v0;
}

fn random_int(seed: ptr<function, u32>) -> u32 {
    // LCG values from Numerical Recipes
    *seed = u32(1664525) * *seed + u32(1013904223);
    return *seed;
}

fn random_float(seed: ptr<function, u32>) -> f32 {
    //// Float version using bitmask from Numerical Recipes
    //const uint one = 0x3f800000;
    //const uint msk = 0x007fffff;
    //return uintBitsToFloat(one | (msk & (RandomInt(seed) >> 9))) - 1;

    // Faster version from NVIDIA examples; quality good enough for our use case.
    return (f32(random_int(seed) & 0x00FFFFFF) / f32(0x01000000));
}

fn random_in_unit_disk(seed: ptr<function, u32>) -> vec2<f32> {
    loop {
        let p = 2.0 * vec2(random_float(seed), random_float(seed)) - 1.0;
        if (dot(p, p) < 1) {
            return p;
        }
    }

    return vec2(0.0);
}