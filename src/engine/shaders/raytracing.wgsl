#import bevy_render::view::View

struct Camera {
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

struct Object {
    index: u32,
    material_id: u32,
}

const MATERIAL_MODEL_LAMBERTIAN: u32 = 0;
const MATERIAL_MODEL_METALLIC: u32 = 1;
const MATERIAL_MODEL_DIELECTRIC: u32 = 2;
const _UNUSED_MATERIAL_MODEL_ISOTROPIC: u32 = 3;
const MATERIAL_MODEL_DIFFUSE_LIGHT: u32 = 4;

struct Material {
    diffuse: vec4<f32>,
    _unused_texture: i32,
    fuzziness: f32,
    refraction_index: f32,
    material_model: u32,
}

struct HitDesc {
    color: vec3<f32>,
    scatter_direction: vec3<f32>,
    scatter: bool,
    albedo: vec3<f32>,
}

const RAY_T_MIN = 0.01f;
const RAY_T_MAX = 100000.0f;

const RAY_NO_CULL = 0xFFu;

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var<storage, read> indices: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> materials: array<Material>;
@group(0) @binding(6) var<storage, read> material_map: array<u32>;

@group(1) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(1) var view_output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var<uniform> light: Light;
@group(1) @binding(3) var<uniform> view: View;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.viewport.zw)) {
        return;
    }

    var pixel_color = vec4(0.0);
    var ray_seed = init_random_seed(init_random_seed(global_id.x, global_id.y), camera.samples * camera.bounces * view.frame_count);
    var pixel_seed = init_random_seed(camera.samples * camera.bounces, camera.samples);

    for (var i = u32(0); i < camera.samples; i++) {
        var jitter = vec2(0.5);
        if (i > 0) {
            jitter = vec2(random_float(&pixel_seed), random_float(&pixel_seed));
        }
        let pixel_center = vec2<f32>(global_id.xy) + jitter;
        let in_uv = pixel_center / vec2(view.viewport.zw);
        let d = in_uv * 2.0 - 1.0;

        let offset = camera.aperture / 2.0 * random_in_unit_disk(&ray_seed);
        var origin = view.world_position;
        let camera_target = view.world_from_clip * vec4(d.x, -d.y, 1.0, 1.0);
        var direction = normalize((camera_target.xyz / camera_target.w) - origin);
        var b = u32(0);

        var accumulated_light = vec3(0.0);
        var throughput = vec3(1.0);

        loop {
            if (b == camera.bounces) {
                break;
            }

            let hit = trace_ray(origin, direction, 0.001, 10000.0, RAY_FLAG_CULL_BACK_FACING);

            var scatter = vec4(0.0);
            if hit.kind != RAY_QUERY_INTERSECTION_NONE {
                scatter = closest_hit(hit, &ray_seed, origin, direction, &accumulated_light, &throughput);
            } else {
                miss(hit, origin, direction, &accumulated_light, &throughput);
            }

            if (scatter.w > 0.0) {
                origin = origin + hit.t * direction;
                direction = scatter.xyz;
            } else {
                break;
            }

            let random = random_float(&ray_seed);
            if (random >= max(throughput.r, max(throughput.g, throughput.b))) {
                break;
            }

            b += 1;
        }

        pixel_color += vec4(accumulated_light, 1.0);
    }

    pixel_color = pixel_color / f32(camera.samples);

    textureStore(view_output, global_id.xy, pixel_color);
}

fn trace_ray(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32) -> RayIntersection {
    let ray = RayDesc(ray_flag, RAY_NO_CULL, ray_t_min, ray_t_max, ray_origin, ray_direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, ray);
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq);
}

fn closest_hit(
    hit: RayIntersection, seed: ptr<function, u32>, origin: vec3<f32>, direction: vec3<f32>,
    accumulated_light: ptr<function, vec3<f32>>, throughput: ptr<function, vec3<f32>>
) -> vec4<f32> {
    let barycentrics = vec3(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics.x, hit.barycentrics.y);

    let object = objects[hit.instance_custom_data];
    let material = materials[material_map[object.material_id + hit.primitive_index]];
    let index = indices[object.index + hit.primitive_index];
    let n0 = normals[index.x].xyz;
    let n1 = normals[index.y].xyz;
    let n2 = normals[index.z].xyz;

    let normal = mat3x3(n0, n1, n2) * barycentrics;
    let world_normal = normalize(mat3x3(hit.object_to_world[0].xyz, hit.object_to_world[1].xyz, hit.object_to_world[2].xyz) * normal);

    var hit_desc = scatter_fn(material, hit.t, seed, world_normal, direction);

    *accumulated_light += hit_desc.color * *throughput;

    if (material.material_model == MATERIAL_MODEL_LAMBERTIAN) {
        let hit_point = origin + hit.t * direction;
        let light_coefficient = max(light.ambient.y * dot(-light.direction.xyz, world_normal), light.ambient.x);

        if (light_coefficient > 0.0) {
            let shadow_origin = hit_point + world_normal * 0.001;
            let shadow_direction = -light.direction.xyz;
            let flags = RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_CULL_NO_OPAQUE | RAY_FLAG_CULL_BACK_FACING;
            let shadow_hit = trace_ray(shadow_origin, shadow_direction, 0.001, 10000.0, flags);

            if (shadow_hit.kind == RAY_QUERY_INTERSECTION_NONE) {
                let direct_light = hit_desc.albedo * light_coefficient;
                *accumulated_light += direct_light * *throughput;
            }
        }
    }

    *throughput *= hit_desc.albedo;
    var scatter = 0.0;
    if (hit_desc.scatter) {
        scatter = 1.0;
    }

    return vec4(hit_desc.scatter_direction, scatter);
}

fn miss(
    hit: RayIntersection, origin: vec3<f32>, direction: vec3<f32>,
    accumulated_light: ptr<function, vec3<f32>>, throughput: ptr<function, vec3<f32>>
) {
    let color = light.sky_color.rgb;

    *accumulated_light += light.sky_color.rgb * *throughput;
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
        if (dot(p, p) < 1.0) {
            return p;
        }
    }

    return vec2(0.0);
}

fn random_in_unit_sphere(seed: ptr<function, u32>) -> vec3<f32> {
    loop {
        let p = 2.0 * vec3(random_float(seed), random_float(seed), random_float(seed)) - 1.0;
        if (dot(p, p) < 1.0) {
            return p;
        }
    }

    return normalize(vec3(1.0));
}

fn schlick(cosine: f32, refraction_index: f32) -> f32 {
    var r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 *= r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

fn scatter_lambertian(material: Material, t: f32, seed: ptr<function, u32>, normal: vec3<f32>, direction: vec3<f32>) -> HitDesc {
    let scatter = dot(direction, normal) < 0.0;
    let color = material.diffuse.rgb;
    let scatter_direction = normal + random_in_unit_sphere(seed);

    return HitDesc(vec3(0.0), normalize(scatter_direction), scatter, color);
}

fn scatter_metallic(material: Material, t: f32, seed: ptr<function, u32>, normal: vec3<f32>, direction: vec3<f32>) -> HitDesc {
    let reflected = reflect(direction, normal);
    let scatter = dot(reflected, normal) > 0.0;
    let color = material.diffuse.rgb;
    let scatter_direction = reflected + material.fuzziness * random_in_unit_sphere(seed);

    return HitDesc(vec3(0.0), normalize(scatter_direction), scatter, color);
}

fn scatter_dielectric(material: Material, t: f32, seed: ptr<function, u32>, normal: vec3<f32>, direction: vec3<f32>) -> HitDesc {
    let scatter = true;
    let dot_value = dot(direction, normal);

    var outward_normal = normal;
    if (dot_value > 0.0) {
        outward_normal = -normal;
    }

    var ni_over_nt = 1.0 / material.refraction_index;
    if (dot_value > 0.0) {
        ni_over_nt = material.refraction_index;
    }

    var cosine = -dot_value;
    if (dot_value > 0.0) {
        cosine = material.refraction_index * dot_value;
    }

    let refracted = refract(direction, outward_normal, ni_over_nt);
    var reflect_probability = 1.0;
    if (any(refracted != vec3(0.0))) {
        reflect_probability = schlick(cosine, material.refraction_index);
    }

    let color = material.diffuse.rgb;

    var scatter_direction = refracted;
    if (random_float(seed) < reflect_probability) {
        scatter_direction = reflect(direction, normal);
    }

    return HitDesc(vec3(0.0), normalize(scatter_direction), scatter, color);
}

fn scatter_diffuse_light(material: Material, t: f32, seed: ptr<function, u32>) -> HitDesc {
    let color = material.diffuse.rgb;

    return HitDesc(color, vec3(0.0), false, vec3(0.0));
}

fn scatter_fn(material: Material, t: f32, seed: ptr<function, u32>, normal: vec3<f32>, direction: vec3<f32>) -> HitDesc {
    switch (material.material_model) {
        case 0: {
            return scatter_lambertian(material, t, seed, normal, direction);
        }

        case 1: {
            return scatter_metallic(material, t, seed, normal, direction);
        }

        case 2: {
            return scatter_dielectric(material, t, seed, normal, direction);
        }

        case 4: {
            return scatter_diffuse_light(material, t, seed);
        }

        default: {
            return HitDesc(vec3(1.0, 0.0, 1.0), vec3(0.0), false, vec3(0.0));
        }
    }
}