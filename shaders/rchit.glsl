#version 460
#extension GL_EXT_ray_tracing: require
#extension GL_EXT_ray_tracing: require
#extension GL_EXT_nonuniform_qualifier: enable
#extension GL_EXT_scalar_block_layout: enable
#extension GL_GOOGLE_include_directive: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_buffer_reference2: require

hitAttributeEXT vec2 attribs;

layout (location = 0) rayPayloadInEXT vec3 hit_value;

struct Voxel {
    vec3 center;
    vec3 color;
};

layout (set = 2, binding = 0) buffer voxels {
    Voxel allVoxels[];
};

void main() {
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    Voxel voxel = allVoxels[gl_PrimitiveID];
    vec3 normal = abs(normalize(worldPos - voxel.center));

    hit_value = allVoxels[gl_PrimitiveID].color;
}