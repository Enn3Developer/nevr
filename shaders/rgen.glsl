#version 460
#extension GL_EXT_ray_tracing: require
#extension GL_EXT_ray_tracing: require
#extension GL_EXT_nonuniform_qualifier: enable
#extension GL_EXT_scalar_block_layout: enable
#extension GL_GOOGLE_include_directive: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_buffer_reference2: require

#include "random.glsl"

const int NBSAMPLES = 10;

layout (location = 0) rayPayloadEXT vec3 hit_value;
layout (set = 0, binding = 0) uniform accelerationStructureEXT top_level_as;
layout (set = 0, binding = 1) uniform Camera {
    mat4 view_proj;    // Camera view * projection
    mat4 view_inverse; // Camera inverse view matrix
    mat4 proj_inverse; // Camera inverse projection matrix
} camera;
layout (set = 1, binding = 0, rgba32f) uniform image2D image;

void main() {
    uint seed = tea(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x, 0);

    vec3 final_color = vec3(0);

    for (int i = 0; i < NBSAMPLES; i++) {
        float r1 = rnd(seed);
        float r2 = rnd(seed);
        seed = tea(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x, seed);

        vec2 jitter = i == 0 ? vec2(0.5) : vec2(r1, r2);

        const vec2 pixel_center = vec2(gl_LaunchIDEXT.xy) + jitter;
        const vec2 in_uv = pixel_center / vec2(gl_LaunchSizeEXT.xy);
        vec2 d = in_uv * 2.0 - 1.0;

        vec4 origin = camera.view_inverse * vec4(1, 1, 1, 1);
        vec4 target = camera.proj_inverse * vec4(d.x, d.y, 1, 1);
        vec4 direction = camera.view_inverse * vec4(normalize(target.xyz), 0);

        uint ray_flags = gl_RayFlagsOpaqueEXT;
        float t_min = 0.001;
        float t_max = 10000.0;

        traceRayEXT(
            top_level_as, // acceleration structure
            ray_flags, // rayFlags
            0xFF, // cullMask
            0, // sbtRecordOffset
            0, // sbtRecordStride
            0, // missIndex
            origin.xyz, // ray origin
            t_min, // ray min range
            direction.xyz, // ray direction
            t_max, // ray max range
            0);            // payload (location = 0)

        final_color += hit_value;
    }

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(final_color / NBSAMPLES, 1.0));
}