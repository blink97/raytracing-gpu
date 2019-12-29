#ifndef AABB_H
# define AABB_H

# include <cuda_runtime.h>

# include "scene.h"

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
 *
 * NOTE: this function is the only one expected
 * to be called by the user, all other functions
 * are only exported for benchamrking purpose.
 */
void compute_bounding_box(const struct scene *const scene, struct AABB *aabbs);

/*
 * All functions exported under this are only present
 * so that they can be benchmarked together.
 */

__global__ void object_compute_bounding_box(const struct scene *const scene, struct AABB *aabbs);

#endif /* !AABB_H */