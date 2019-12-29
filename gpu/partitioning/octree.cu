#include "octree.h"

__device__ __forceinline__ float atomicMinFloat(float *addr, float value)
{
  float old = ((value >= 0)
    ? __int_as_float(atomicMin((int *)addr, __float_as_int(value)))
    : __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value))));

  return old;
}

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
  float old = ((value >= 0)
    ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
    : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value))));

    return old;
}


__global__ void find_scene_scale_basic(
  const struct AABB *const aabb,
  size_t nb_objects,
  struct AABB *resulting_scale)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nb_objects) return; // Nothing to do here

  atomicMinFloat(&resulting_scale->min.x, aabb[index].min.x);
  atomicMinFloat(&resulting_scale->min.y, aabb[index].min.y);
  atomicMinFloat(&resulting_scale->min.z, aabb[index].min.z);

  atomicMaxFloat(&resulting_scale->max.x, aabb[index].max.x);
  atomicMaxFloat(&resulting_scale->max.y, aabb[index].max.y);
  atomicMaxFloat(&resulting_scale->max.z, aabb[index].max.z);
}

__global__ void find_scene_scale_shared(
  const struct AABB *const aabb,
  size_t nb_objects,
  struct AABB *resulting_scale)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nb_objects) return; // Nothing to do here

  __shared__ struct AABB shared_scale;

  atomicMinFloat(&shared_scale.min.x, aabb[index].min.x);
  atomicMinFloat(&shared_scale.min.y, aabb[index].min.y);
  atomicMinFloat(&shared_scale.min.z, aabb[index].min.z);

  atomicMaxFloat(&shared_scale.max.x, aabb[index].max.x);
  atomicMaxFloat(&shared_scale.max.y, aabb[index].max.y);
  atomicMaxFloat(&shared_scale.max.z, aabb[index].max.z);

  // Make sure that all the thread all computed the correct partial scale
  // before doing it at the global level.
  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicMinFloat(&resulting_scale->min.x, shared_scale.min.x);
    atomicMinFloat(&resulting_scale->min.y, shared_scale.min.y);
    atomicMinFloat(&resulting_scale->min.z, shared_scale.min.z);

    atomicMaxFloat(&resulting_scale->max.x, shared_scale.max.x);
    atomicMaxFloat(&resulting_scale->max.y, shared_scale.max.y);
    atomicMaxFloat(&resulting_scale->max.z, shared_scale.max.z);
  }
}

void create_octree(
  struct scene *scene,
  struct AABB *aabb,
  struct octree **octree)
{
  // Find the maximun and the minimum of the whole scene.
  // Scale the aabb to fit in the [0-1] space

  // For a fixed depth, let all objects place themself in the octree.
  // To do that, use 3bits per level to say where they are stored.

  // Sort the objects to group together objects at the same level.

  // Create the octree from that.
}