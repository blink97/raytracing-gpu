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
  uint8_t resulting_position_x = position_max_x & (0xFF << (8 - final_level));
  uint8_t resulting_position_y = position_max_y & (0xFF << (8 - final_level));
  uint8_t resulting_position_z = position_max_z & (0xFF << (8 - final_level));

  // Create the resulting position.
  octree_generation_position position = final_level;
  for (int i = 0; i < 8; ++i)
  {
    position <<= 3;
    position |= ((resulting_position_x & (1 << (7 - i))) != 0 ? 1 : 0);
    position |= ((resulting_position_y & (1 << (7 - i))) != 0 ? 2 : 0);
    position |= ((resulting_position_z & (1 << (7 - i))) != 0 ? 4 : 0);
  }

  // And save it
  positions[index] = position;
}


__global__ void single_thread_bubble_argsort(
  octree_generation_position *positions,
  size_t *indexes,
  size_t nb_objects)
{
  if (blockIdx.x * blockDim.x + threadIdx.x > 1)
    return; // Nothing to do here, sort is single thread.

  // First, create the range.
  for (size_t i = 0; i < nb_objects; ++i)
    indexes[i] = i;

  // Then bubble sort the way out of the array.
  for (size_t i = nb_objects - 1; i > 0; --i)
  {
    for (size_t j = 0; j < i; ++j)
    {
      if (positions[j + 1] < positions[j])
      {
        size_t temp_index = indexes[j];
        octree_generation_position temp_position = positions[j];

        indexes[j] = indexes[j + 1];
        positions[j] = positions[j + 1];

        indexes[j + 1] = temp_index;
        positions[j + 1] = temp_position;
      }
    }
  }
}

__global__ void nodes_difference_array(
  const octree_generation_position *const sorted_positions,
  size_t *node_differences,
  size_t nb_objects)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_objects) return; // Nothing to do here

  octree_generation_position current = sorted_positions[index];
  uint8_t current_level = ((current & 0xFF000000) >> 24);

  size_t diff;
  if (index == 0)
  {// The first node creates everything.
    diff = current_level + 1/* Include the root nodes */;
  }
  else
  {// The next nodes only created was is needed compared to the previous one.
    octree_generation_position previous = sorted_positions[index - 1];
    uint8_t previous_level = ((previous & 0xFF000000) >> 24);

    uint8_t min_level;
    uint8_t max_level;
    if (previous_level < current_level)
    {
      min_level = previous_level;
      max_level = current_level;
    }
    else
    {
      min_level = current_level;
      max_level = previous_level;
    }

    diff = max_level;// Don't include the root node, it already was included.

    uint8_t actual_test_level = 0;
    for (; actual_test_level < min_level; ++actual_test_level)
    {
      if (((previous >> (3 * (7 - actual_test_level))) & 0x7) !=
          ((current >> (3 * (7 - actual_test_level))) & 0x7))
      {// Current level and previous level is different
        break;
      }
    }

    // Remove the common level nodes in it.
    diff -= actual_test_level;
  }

  node_differences[index] = diff;
}

__global__ void single_thread_nodes_difference_to_prefix_array(
  size_t *nodes_differences,
  size_t nb_objects)
{
  if (blockIdx.x * blockDim.x + threadIdx.x > 1)
    return; // Nothing to do here, prefix array is single thread.

  size_t previous = 0;
  for (size_t i = 0; i < nb_objects; ++i)
  {
    previous = nodes_differences[i] + previous;
    nodes_differences[i] = previous;
  }
}

__global__ void create_octree(
  const octree_generation_position *const sorted_positions,
  const size_t *const sorted_indexes,
  const size_t *const node_differences,
  size_t nb_objects,
  struct octree **resulting_octree)
{
  // Find the maximun and the minimum of the whole scene.
  // Scale the aabb to fit in the [0-1] space

  // For a fixed depth, let all objects place themself in the octree.
  // To do that, use 3bits per level to say where they are stored.

  // Sort the objects to group together objects at the same level.

  // Create the octree from that.
}