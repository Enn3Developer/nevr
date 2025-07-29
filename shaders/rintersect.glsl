#version 460
#extension GL_EXT_ray_tracing: require
#extension GL_EXT_nonuniform_qualifier: enable
#extension GL_EXT_scalar_block_layout: enable
#extension GL_GOOGLE_include_directive: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_buffer_reference2: require

hitAttributeEXT vec2 attribs;

struct Voxel {
    vec3 minimum;
    vec3 maximum;
    uint material_id;
};

struct Ray
{
    vec3 origin;
    vec3 direction;
};

layout (set = 2, binding = 0) buffer voxels {
    Voxel allVoxels[];
};

// compute the near and far intersections of the cube (stored in the x and y components) using the slab method
// no intersection means vec.x > vec.y (really tNear > tFar)
vec2 intersectAABB(Ray ray, vec3 boxMin, vec3 boxMax) {
    vec3 tMin = (boxMin - ray.origin) / ray.direction;
    vec3 tMax = (boxMax - ray.origin) / ray.direction;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
};

void main() {
    Ray ray;
    ray.origin = gl_WorldRayOriginEXT;
    ray.direction = gl_WorldRayDirectionEXT;

    Voxel voxel = allVoxels[gl_InstanceCustomIndexEXT * 8192 + gl_PrimitiveID];

    vec2 t = intersectAABB(ray, voxel.minimum, voxel.maximum);

    if (t.x <= t.y) {
        attribs = t;
        reportIntersectionEXT(t.x, 1);
    }
}