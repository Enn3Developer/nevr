#version 460
#extension GL_EXT_ray_tracing: require
#extension GL_EXT_nonuniform_qualifier: enable
#extension GL_EXT_scalar_block_layout: enable
#extension GL_GOOGLE_include_directive: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_buffer_reference2: require

#include "random.glsl"
#include "raycommon.glsl"

layout (location = 0) rayPayloadEXT RayPayload Ray;
layout (set = 0, binding = 0) uniform accelerationStructureEXT top_level_as;
layout (set = 0, binding = 1) uniform Camera {
    mat4 view_proj;    // Camera view * projection
    mat4 view_inverse; // Camera inverse view matrix
    mat4 proj_inverse; // Camera inverse projection matrix
    float aperture;
    float focusDistance;
    uint samples;
    uint bounces;
    uint frame;
} camera;
layout (set = 1, binding = 0, rgba32f) uniform image2D image;
layout (set = 3, binding = 1) uniform Light {
    vec4 ambient_light;
    vec4 light_direction;
} light;

void main() {
    Ray.RandomSeed = InitRandomSeed(InitRandomSeed(gl_LaunchIDEXT.x, gl_LaunchIDEXT.y), camera.frame * camera.samples * camera.bounces);
    uint pixelRandomSeed = InitRandomSeed(camera.frame * camera.samples * camera.bounces, camera.frame * camera.samples);

    vec3 pixelColor = vec3(0);

    for (int i = 0; i < camera.samples; i++) {
        float r1 = RandomFloat(pixelRandomSeed);
        float r2 = RandomFloat(pixelRandomSeed);

        vec2 jitter = i == 0 ? vec2(0.5) : vec2(r1, r2);

        const vec2 pixel_center = vec2(gl_LaunchIDEXT.xy) + jitter;
        const vec2 in_uv = pixel_center / vec2(gl_LaunchSizeEXT.xy);
        vec2 d = in_uv * 2.0 - 1.0;

        vec2 offset = camera.aperture / 2 * RandomInUnitDisk(Ray.RandomSeed);
        vec4 origin = camera.view_inverse * vec4(offset, 0, 1);
        vec4 target = camera.proj_inverse * vec4(d.x, d.y, 1, 1);
        vec4 direction = camera.view_inverse * vec4(normalize(target.xyz * camera.focusDistance - vec3(offset, 0)), 0);
        vec3 rayColor = vec3(1);

        uint ray_flags = gl_RayFlagsOpaqueEXT;
        float t_min = 0.01;
        float t_max = 10000.0;

        for (int b = 0; b <= camera.bounces; b++) {
            if (b == camera.bounces) {
                rayColor = vec3(0);
                break;
            }

            traceRayEXT(
                top_level_as,
                ray_flags,
                0xFF,
                0, // sbtOffset
                0, // sbtStride
                0, // missIndex
                origin.xyz,
                t_min,
                direction.xyz,
                t_max,
                0
            );

            vec3 hitColor = Ray.ColorAndDistance.rgb;
            float t = Ray.ColorAndDistance.w;
            bool isScattered = Ray.ScatterDirection.w > 0;

            rayColor *= hitColor;

            if (t < 0 || !isScattered)
            {
                break;
            }

            origin = origin + t * direction;
            direction = vec4(Ray.ScatterDirection.xyz, 0);
        }

        pixelColor += rayColor;
    }

    pixelColor = pixelColor / camera.samples;
    pixelColor = sqrt(pixelColor);

    if (false) {
        vec3 old_color = imageLoad(image, ivec2(gl_LaunchIDEXT.xy)).xyz;
        vec3 color = (old_color * camera.frame + pixelColor) / (camera.frame + 1);
        imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(color, 1.0));
    } else {
        imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(pixelColor, 1.0));
    }
}