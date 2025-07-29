#version 460
#extension GL_EXT_ray_tracing: require
#extension GL_EXT_nonuniform_qualifier: enable
#extension GL_EXT_scalar_block_layout: enable
#extension GL_GOOGLE_include_directive: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_buffer_reference2: require

#include "random.glsl"
#include "raycommon.glsl"
#include "material.glsl"
#include "scatter.glsl"

hitAttributeEXT vec2 attribs;

layout (location = 0) rayPayloadInEXT RayPayload Ray;
layout (location = 1) rayPayloadEXT bool isShadowed;
layout (set = 0, binding = 0) uniform accelerationStructureEXT top_level_as;

struct Voxel {
    vec3 minimum;
    vec3 maximum;
    uint material_id;
};

layout (set = 2, binding = 0) buffer voxels {
    Voxel allVoxels[];
};

layout (set = 2, binding = 1) buffer materials {
    Material allMaterials[];
};

layout (set = 3, binding = 1) uniform Light {
    vec4 ambient_light;
    vec4 light_direction;
} light;

vec3 compute_normal(vec3 hitPos, vec3 aabbMin, vec3 aabbMax) {
    vec3 center = (aabbMin + aabbMax) * 0.5;

    vec3 direction = hitPos - center;
    vec3 divisor = (aabbMin - aabbMax) * 0.5;
    float bias = 1.0001;
    vec3 normal = vec3(ivec3(direction / abs(divisor) * bias));

    return normalize(normal);
}

void main() {
    vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    Voxel voxel = allVoxels[gl_InstanceCustomIndexEXT * 8192 + gl_PrimitiveID];
    Material material = allMaterials[voxel.material_id];
    vec3 normal = compute_normal(hitPos, voxel.minimum, voxel.maximum);

    float light_coefficient = max(dot(-light.light_direction.xyz, normal), light.ambient_light.r);

    Ray = Scatter(material, gl_WorldRayDirectionEXT, normal, vec2(0), gl_HitTEXT, Ray.RandomSeed);
    Ray.ColorAndDistance.rgb = Ray.ColorAndDistance.rgb * light_coefficient;

    if (dot(normal, -light.light_direction.xyz) > 0.0) {
        float t_min = 0.001;
        float t_max = 10000.0;
        vec3 origin = hitPos;
        vec3 rayDir = -light.light_direction.xyz;
        uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
        isShadowed = true;

        traceRayEXT(
            top_level_as, // acceleration structure
            flags, // rayFlags
            0xFF, // cullMask
            0, // sbtRecordOffset
            0, // sbtRecordStride
            1, // missIndex
            origin, // ray origin
            t_min, // ray min range
            rayDir, // ray direction
            t_max, // ray max range
            1            // payload (location = 1)
        );

        if (isShadowed) {
            Ray.ColorAndDistance.rgb = Ray.ColorAndDistance.rgb * light.ambient_light.rgb;
        }
    }
}