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
  const struct AABB *const aabbs,
  size_t nb_objects,
  struct AABB *resulting_scale)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nb_objects) return; // Nothing to do here

  atomicMinFloat(&resulting_scale->min.x, aabbs[index].min.x);
  atomicMinFloat(&resulting_scale->min.y, aabbs[index].min.y);
  atomicMinFloat(&resulting_scale->min.z, aabbs[index].min.z);

  atomicMaxFloat(&resulting_scale->max.x, aabbs[index].max.x);
  atomicMaxFloat(&resulting_scale->max.y, aabbs[index].max.y);
  atomicMaxFloat(&resulting_scale->max.z, aabbs[index].max.z);
}

__global__ void find_scene_scale_shared(
  const struct AABB *const aabbs,
  size_t nb_objects,
  struct AABB *resulting_scale)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nb_objects) return; // Nothing to do here

  __shared__ struct AABB shared_scale;

  atomicMinFloat(&shared_scale.min.x, aabbs[index].min.x);
  atomicMinFloat(&shared_scale.min.y, aabbs[index].min.y);
  atomicMinFloat(&shared_scale.min.z, aabbs[index].min.z);

  atomicMaxFloat(&shared_scale.max.x, aabbs[index].max.x);
  atomicMaxFloat(&shared_scale.max.y, aabbs[index].max.y);
  atomicMaxFloat(&shared_scale.max.z, aabbs[index].max.z);

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

__device__ void scale_position(vector3 *position, const struct AABB *const scale)
{
  position->x = (position->x + scale->min.x) / (scale->max.x - scale->min.x);
  position->y = (position->y + scale->min.y) / (scale->max.y - scale->min.y);
  position->z = (position->z + scale->min.z) / (scale->max.z - scale->min.z);
}


#define POINT_POSITION(value) (min((uint16_t)((value) * 256), 255))

// Get the level associated with the object
// (the octree node that can contains both min and max value)
__device__ void object_level(uint8_t min, uint8_t max, uint8_t *level)
{
  uint8_t current_level = 0;

  // Trying to find it's place from top to bottom:
  // If a level is accepted, it goes to the next one.
  while (current_level < 8 && (min & (1 << (7 - current_level))) == (max & (1 << (7 - current_level))))
  {
    ++current_level;
  }

  *level = current_level;
}

__global__ void position_object(
  const struct AABB *const aabbs,
  const struct AABB *const scale,
  octree_generation_position *positions,
  size_t nb_objects)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nb_objects) return; // Nothing to do here

  struct AABB current_aabb = aabbs[index];

  // Scale the AABB so that it is in the [0-1] cube
  scale_position(&current_aabb.min, scale);
  scale_position(&current_aabb.max, scale);

  // Start to find it's position.
  // To do so, lay a fixed grid, of the biggest depth and lay the number in it.
  // Doing so allows to have the correct value.
  // (1 if above center, 0 if under, whatever is the depth).

  uint8_t position_min_x = POINT_POSITION(current_aabb.min.x);
  uint8_t position_min_y = POINT_POSITION(current_aabb.min.y);
  uint8_t position_min_z = POINT_POSITION(current_aabb.min.z);

  uint8_t position_max_x = POINT_POSITION(current_aabb.max.x);
  uint8_t position_max_y = POINT_POSITION(current_aabb.max.y);
  uint8_t position_max_z = POINT_POSITION(current_aabb.max.z);

  uint8_t level_x, level_y, level_z;

  object_level(position_min_x, position_max_x, &level_x);
  object_level(position_min_y, position_max_y, &level_y);
  object_level(position_min_z, position_max_z, &level_z);

  // The final level is the top level in all axes.
  uint8_t final_level = min(min(level_x, level_y), level_z);

  // Compute the final position, position_min is used,
  // but position_min and position_max point to the same thing,
  // as the level is how many common bits they have in common.
  uint8_t resulting_position_x = position_min_x | (0xFF << (8 - final_level));
  uint8_t resulting_position_y = position_min_y | (0xFF << (8 - final_level));
  uint8_t resulting_position_z = position_min_z | (0xFF << (8 - final_level));

  // Create the resulting position.
  octree_generation_position position = final_level;
  for (int i = 0; i < 8; ++i)
  {
    position <<= 3;
    position |= (resulting_position_x & (1 << (7 - i)) ? 4 : 0);
    position |= (resulting_position_y & (1 << (7 - i)) ? 2 : 0);
    position |= (resulting_position_z & (1 << (7 - i)) ? 1 : 0);
  }

  // And save it
  positions[index] = position;
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