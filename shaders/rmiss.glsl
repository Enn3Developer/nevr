#version 460
#extension GL_EXT_ray_tracing: require
#extension GL_EXT_nonuniform_qualifier: enable
#extension GL_EXT_scalar_block_layout: enable
#extension GL_GOOGLE_include_directive: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_buffer_reference2: require

#include "raycommon.glsl"

layout (location = 0) rayPayloadInEXT RayPayload Ray;
layout (set = 3, binding = 0) buffer skyColor {
    vec3 sky_color;
};

void main() {
    // Sky color
    const float t = 0.5 * (normalize(gl_WorldRayDirectionEXT).y + 1);
    const vec3 skyColor = mix(vec3(1.0), vec3(0.5, 0.7, 1.0), t);

    Ray.ColorAndDistance = vec4(sky_color, -1);
}