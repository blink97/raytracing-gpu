#include "aabb.h"

/*
 * Compute the axis-aligned bounding box,
 * but doing it per objects: each thread is reponsible
 * to compute the aabb of only one objects (their own).
 */
__global__ void object_compute_bounding_box(const struct scene *const scene, struct AABB *aabbs)
{
    size_t object_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (object_index >= scene->object_count) return; // Nothing to do here

    // Create unbelievable point to replace them the first time.
    vector3 min_point = make_float3(1000, 1000, 1000);
    vector3 max_point = make_float3(-1000, -1000, -1000);

    const struct object *const current_object = &scene->objects[object_index];

    for (uint32_t i = 0; i < current_object->triangle_count; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            min_point.x = min(min_point.x, get_vertex(current_object->triangles, i)[j].x);
            min_point.y = min(min_point.y, get_vertex(current_object->triangles, i)[j].y);
            min_point.z = min(min_point.z, get_vertex(current_object->triangles, i)[j].z);

            max_point.x = max(max_point.x, get_vertex(current_object->triangles, i)[j].x);
            max_point.y = max(max_point.y, get_vertex(current_object->triangles, i)[j].y);
            max_point.z = max(max_point.z, get_vertex(current_object->triangles, i)[j].z);
        }
    }

    aabbs[object_index].min = min_point;
    aabbs[object_index].max = max_point;
}

void compute_bounding_box(const struct scene *const scene, struct AABB *aabbs)
{
    // Compute the boundign box per objects.
    dim3 threadsPerBlock(32);
    dim3 numBlocks(ceil(scene->object_count * 1.0 / threadsPerBlock.x));

    object_compute_bounding_box<<<numBlocks, threadsPerBlock>>>(scene, aabbs);
}