#ifndef AABB_H
# define AABB_H

# include <cuda_runtime.h>

# include "scene.h"
# include "ray.h"

/*
 * Axis aligned bounding box.
 * Is used to have an approximation of objects shape
 * without having to look for all the vertex in it.
 *
 * Can be used as it, but is intended to be used
 * as a base for the quadtree/octree.
 */
struct AABB {
  vector3 min;
  vector3 max;
};

/*
 * Compute the bounding box of the given scene.
 * The given AABB array must have the same size
 * as the number of objects in the scene, as well
 * as being a GPU memory address.
 */
void compute_bounding_box(const struct scene *const scene, struct AABB *aabbs);

__global__ void fill_object_triangle_count(
  const struct object *const objects,
  size_t *objects_triangles_count,
  size_t size
);

__global__ void object_compute_bounding_box(const struct scene *const scene, struct AABB *aabbs);

__host__ __device__ bool hit_aabb(const struct AABB *const aabb, const struct ray *const ray);

#endif /* !AABB_H */