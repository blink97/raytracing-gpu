#include "aabb.h"
#include "prefix_sum.h"
#include "utils.h"

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

/* Layout dependent code */
#   if defined(LAYOUT_AOS)

__device__ vector3* get_triangle_vertex(const struct scene *scene, uint32_t triangle_index)
{
  return scene->objects_data.vertex_and_normal + triangle_index * 6;
}

#   elif defined(LAYOUT_SOA)

__device__ vector3* get_triangle_vertex(const struct scene *scene, uint32_t triangle_index)
{
  return scene->objects_data.vertex + triangle_index * 3;
}

#   endif
/* End of layout dependent code */

/* Layout dependent code */
# if !defined(LAYOUT_FRAGMENTED)// LAYOUT_AOS || LAYOUT_SOA

#  define AABB_TRIANGLE_BLOCK_SIZE 128

__global__ void fill_object_triangle_count(
  const struct object *const objects,
  size_t *objects_triangles_count,
  size_t size)
{
  size_t object_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (object_index >= size)
    return; // Nothing to do here

  objects_triangles_count[object_index] = objects[object_index].triangle_count;
}


__global__ void triangle_compute_bounding_box(
  const struct scene *const scene,
  const size_t *const objects_triangles_count,
  struct AABB *aabbs,
  size_t size)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= objects_triangles_count[size - 1])
    return; // Nothing to do here


  // Find the objects containing the current triangle
  // Use a binary search, making first steps
  // essentially free: all triangles will touch the same memory cells
  // allowing for coalesence reading.

  size_t object_index = binary_search(objects_triangles_count, size, index);
  size_t first_triangle_index = (object_index == 0
    ? 0
    : objects_triangles_count[object_index - 1]
  );
  size_t first_triangle_thread_index = (first_triangle_index < index - threadIdx.x
      ? index - threadIdx.x// First triangle index is in another block, only taking the first of the block.
      : first_triangle_index// First triangle index is in this block.
  ) % AABB_TRIANGLE_BLOCK_SIZE;

  __shared__ struct AABB shared_aabb[AABB_TRIANGLE_BLOCK_SIZE];

  vector3 *vertex = get_triangle_vertex(scene, index);

  if (threadIdx.x == first_triangle_thread_index)
  {// The first triangles of the object (per thread) set a starting value.
    shared_aabb[first_triangle_thread_index].min = vertex[0];
    shared_aabb[first_triangle_thread_index].max = vertex[0];
  }

  if (index == first_triangle_index)
  {// The first triangle per object (globally) set a starting value
    aabbs[object_index].min = vertex[0];
    aabbs[object_index].max = vertex[0];
  }

  __syncthreads();

  // Locally perform the min max
  for (uint8_t i = 0; i < 3; ++i)
  {
    atomicMinFloat(&shared_aabb[first_triangle_thread_index].min.x, vertex[i].x);
    atomicMinFloat(&shared_aabb[first_triangle_thread_index].min.y, vertex[i].y);
    atomicMinFloat(&shared_aabb[first_triangle_thread_index].min.z, vertex[i].z);

    atomicMaxFloat(&shared_aabb[first_triangle_thread_index].max.x, vertex[i].x);
    atomicMaxFloat(&shared_aabb[first_triangle_thread_index].max.y, vertex[i].y);
    atomicMaxFloat(&shared_aabb[first_triangle_thread_index].max.z, vertex[i].z);
  }

  __syncthreads();

  // Globally perform the min max
  if (threadIdx.x == first_triangle_thread_index)
  {
    atomicMinFloat(&aabbs[object_index].min.x, shared_aabb[first_triangle_thread_index].min.x);
    atomicMinFloat(&aabbs[object_index].min.y, shared_aabb[first_triangle_thread_index].min.y);
    atomicMinFloat(&aabbs[object_index].min.z, shared_aabb[first_triangle_thread_index].min.z);

    atomicMaxFloat(&aabbs[object_index].max.x, shared_aabb[first_triangle_thread_index].max.x);
    atomicMaxFloat(&aabbs[object_index].max.y, shared_aabb[first_triangle_thread_index].max.y);
    atomicMaxFloat(&aabbs[object_index].max.z, shared_aabb[first_triangle_thread_index].max.z);
  }
}

# endif
/* End of layout dependent code */


void compute_bounding_box(const struct scene *const scene, struct AABB *aabbs)
{
  struct scene CPU_scene;
  cudaMemcpy(&CPU_scene, scene, sizeof(struct scene), cudaMemcpyDefault);

  /* Layout dependent code */
  # if defined(LAYOUT_FRAGMENTED)

  // Can't do any optimisations as the layout is fragmented,
  // so triangles can't be directly accessed.
  dim3 threadsPerBlock(32);
  dim3 numBlocks(ceil(CPU_scene.object_count * 1.0 / threadsPerBlock.x));

  object_compute_bounding_box<<<threadsPerBlock, numBlocks>>>(scene, aabbs);

  # else // LAYOUT_AOS || LAYOUT_SOA

  size_t *objects_triangles_count;
  cudaMalloc(&objects_triangles_count, sizeof(size_t) * CPU_scene.object_count);

  // fill with objects triangle count
  fill_object_triangle_count<<<ceil(CPU_scene.object_count * 1.0 / 128), 128>>>(
    CPU_scene.objects,
    objects_triangles_count,
    CPU_scene.object_count
  );

  // Perform a prefix sum on it so that each triangles
  // knows to which object they belongs.
  shared_prefix_sum(objects_triangles_count, CPU_scene.object_count);

  // Get back the triangle count.
  size_t triangles_count;
  cudaMemcpy(
    &triangles_count,
    objects_triangles_count + CPU_scene.object_count - 1,
    sizeof(size_t), cudaMemcpyDefault
  );

  // Fill the aabbs
  triangle_compute_bounding_box<<<
    ceil(triangles_count * 1.0 / AABB_TRIANGLE_BLOCK_SIZE),
    AABB_TRIANGLE_BLOCK_SIZE
  >>>(scene, objects_triangles_count, aabbs, CPU_scene.object_count);

  cudaFree(objects_triangles_count);

  # endif
  /* End of layout dependent code */
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
