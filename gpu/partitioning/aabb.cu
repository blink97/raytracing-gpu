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

  const struct object *const current_object = scene->objects + object_index;

  for (uint32_t i = 0; i < current_object->triangle_count; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      min_point.x = fmin(min_point.x, get_vertex(current_object->triangles, i)[j].x);
      min_point.y = fmin(min_point.y, get_vertex(current_object->triangles, i)[j].y);
      min_point.z = fmin(min_point.z, get_vertex(current_object->triangles, i)[j].z);

      max_point.x = fmax(max_point.x, get_vertex(current_object->triangles, i)[j].x);
      max_point.y = fmax(max_point.y, get_vertex(current_object->triangles, i)[j].y);
      max_point.z = fmax(max_point.z, get_vertex(current_object->triangles, i)[j].z);
    }
  }

  aabbs[object_index].min = min_point;
  aabbs[object_index].max = max_point;
}


// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
__host__ __device__ bool hit_aabb(const struct AABB *const aabb, const struct ray *const ray)
{
  float tmin, tmax, tymin, tymax, tzmin, tzmax;

  vector3 inv_direction = make_float3(// If == zero, should map to infinity
    1 / ray->direction.x,
    1 / ray->direction.y,
    1 / ray->direction.z
  );

  int sign_x = (inv_direction.x < 0);
  int sign_y = (inv_direction.y < 0);
  int sign_z = (inv_direction.z < 0);

  tmin = ((sign_x ? aabb->max.x : aabb->min.x) - ray->origin.x) * inv_direction.x;
  tmax = ((sign_x ? aabb->min.x : aabb->max.x) - ray->origin.x) * inv_direction.x;
  tymin = ((sign_y ? aabb->max.y : aabb->min.y) - ray->origin.y) * inv_direction.y;
  tymax = ((sign_y ? aabb->min.y : aabb->max.y) - ray->origin.y) * inv_direction.y;

  if ((tmin > tymax) || (tymin > tmax))
    return false;
  if (tymin > tmin)
    tmin = tymin;
  if (tymax < tmax)
    tmax = tymax;

  tzmin = ((sign_z ? aabb->max.z : aabb->min.z) - ray->origin.z) * inv_direction.z;
  tzmax = ((sign_z ? aabb->min.z : aabb->max.z) - ray->origin.z) * inv_direction.z;

  if ((tmin > tzmax) || (tzmin > tmax))
    return false;

  /*
  if (tzmin > tmin)
    tmin = tzmin;
  if (tzmax < tmax)
    tmax = tzmax;
  */

  return true;
}
