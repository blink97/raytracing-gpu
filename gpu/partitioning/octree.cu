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

/*
 * Get the position of a point in one dimension
 * into a virtual octree. The given value must
 * lie in the [0-1] range
 */
__device__ uint8_t get_point_position(float value)
{
  return min((uint16_t)((value) * 256), 255);
}

/*
 * Get the level associated with the given position
 */
__device__ uint8_t get_level(octree_generation_position position)
{
  return position & 0xFF;
}

/*
 * Get the position in the octree node at the given level.
 */
__device__ uint8_t get_level_position(octree_generation_position position, uint8_t level)
{
  return (position >> (8/* Skip the level */ + 3 * (8 - level))) & 0x7;
}


__device__ uint8_t get_common_level(
    octree_generation_position first,
    octree_generation_position second)
{
  uint8_t min_level = min(get_level(first), get_level(second));
  uint8_t common_level = 0;
  for (; common_level < min_level; ++common_level)
  {
    if (get_level_position(first, common_level + 1) != get_level_position(second, common_level + 1))
    {// first and second levels are differents
      break;
    }
  }
  return common_level;
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

  // Set the default value of the scale
  if (threadIdx.x == 0)
    shared_scale = aabbs[index];
  __syncthreads();

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

  uint8_t position_min_x = get_point_position(current_aabb.min.x);
  uint8_t position_min_y = get_point_position(current_aabb.min.y);
  uint8_t position_min_z = get_point_position(current_aabb.min.z);

  uint8_t position_max_x = get_point_position(current_aabb.max.x);
  uint8_t position_max_y = get_point_position(current_aabb.max.y);
  uint8_t position_max_z = get_point_position(current_aabb.max.z);

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
  octree_generation_position position;
  for (int i = 0; i < 8; ++i)
  {
    position <<= 3;
    position |= ((resulting_position_x & (1 << (7 - i))) != 0 ? 1 : 0);
    position |= ((resulting_position_y & (1 << (7 - i))) != 0 ? 2 : 0);
    position |= ((resulting_position_z & (1 << (7 - i))) != 0 ? 4 : 0);
  }
  position = ((position << 8) | final_level);

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
  size_t *nodes_difference,
  size_t nb_objects)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nb_objects) return; // Nothing to do here

  octree_generation_position current = sorted_positions[index];
  uint8_t current_level = get_level(current);

  size_t diff;
  if (index == 0)
  {// The first node creates everything.
    diff = current_level + 1/* Include the root nodes */;
  }
  else
  {// The next nodes only created was is needed compared to the previous one.
    octree_generation_position previous = sorted_positions[index - 1];
    uint8_t previous_level = get_level(previous);

    // Don't include the root node, it already was included.
    // Remove the common level nodes in it.
    uint8_t common_level = get_common_level(previous, current);
    diff = current_level - common_level;
  }

  nodes_difference[index] = diff;
}


__global__ void single_thread_nodes_difference_to_prefix_array(
  size_t *nodes_difference,
  size_t nb_objects)
{
  if (blockIdx.x * blockDim.x + threadIdx.x > 1)
    return; // Nothing to do here, prefix array is single thread.

  size_t previous = 0;
  for (size_t i = 0; i < nb_objects; ++i)
  {
    previous = nodes_difference[i] + previous;
    nodes_difference[i] = previous;
  }
}


__device__ vector3 get_center(uint8_t x, uint8_t y, uint8_t z, uint8_t level, const struct AABB *const scale)
{
  float center_offset = pow(0.5, level) * 0.5;
  vector3 center = {
    .x = ((float)(x / 256.0) + center_offset) * (scale->max.x - scale->min.x) - scale->min.x,
    .y = ((float)(y / 256.0) + center_offset) * (scale->max.y - scale->min.y) - scale->min.y,
    .z = ((float)(z / 256.0) + center_offset) * (scale->max.z - scale->min.z) - scale->min.z
  };
  return center;
}

__global__ void create_octree(
  const octree_generation_position *const sorted_positions,
  const size_t *const nodes_difference,
  size_t nb_objects,
  const struct AABB *const scale,
  struct octree *resulting_octree)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nb_objects) return; // Nothing to do here

  /*
   * As the objects are sorted, only two things must be done:
   * - if the object create a new hierachy
   *    - Set the center of the octree
   *    - set the start range
   *    - Set the parent pointer of the children
   * - if the object end the hierachy (next one create a new hierachy)
   *    - Set the end range
   */


  size_t previous_diff = (index == 0 ? 0 : nodes_difference[index - 1]);
  size_t current_diff = nodes_difference[index];

  if (previous_diff != current_diff)
  {// This object create a new hierachy

    // Get the center of the octree.
    // To do that, perform the inverse trick of get_point_position
    // to get a value in the [0-1[ range, and scale it back to get the center.
    octree_generation_position current_position = sorted_positions[index];
    uint8_t x, y, z, level = get_level(current_position);
    for (uint8_t i = 0; i < 8; ++i)
    {
      uint8_t local_position = get_level_position(current_position, i + 1);
      x = x << 1 | ((local_position & 1) != 0);
      y = y << 1 | ((local_position & 2) != 0);
      z = z << 1 | ((local_position & 4) != 0);
    }

    // The new node have been created, but unused
    // so their start and end must be set.
    for (size_t i = previous_diff + 1/* Skip the already created parent */; i < current_diff; ++i)
    {
      resulting_octree[i - 1].start_index = index;
      resulting_octree[i - 1].end_index = index;
      resulting_octree[i - 1].center = get_center(x, y, z, level - (current_diff - i), scale);
    }

    // Set the starting index.
    resulting_octree[current_diff - 1].start_index = index;
    resulting_octree[current_diff - 1].center = get_center(x, y, z, level, scale);
  }

  if (index + 1 >= nb_objects || nodes_difference[index + 1] != current_diff)
  {// This object end the current hierachy
    resulting_octree[current_diff - 1].end_index = index + 1;
  }

  // As the parent finding research might need to iterate over
  // all previous objects, do it only when all start and end range
  // have been done, so that most object can safely be skipped over.
  __syncthreads();


  if (previous_diff != current_diff)
  {// Create a new hierachy, find parents and set the children
    octree_generation_position current_position = sorted_positions[index];
    size_t bottom_level = get_level(current_position);

    // First, fixes all created nodes, except the last (there is nothing to fix here)
    for (size_t index = previous_diff; index + 1 < current_diff; ++index)
    {
      uint8_t fix_level = bottom_level - (current_diff - (index + 2));
      resulting_octree[index].children[
        get_level_position(current_position, fix_level)
      ] = (resulting_octree + index + 1);
    }

    size_t fix_level = get_level(current_position) - (current_diff - previous_diff);

    // Then fix to highest created node as it's parent is not known.
    // First find it's parent.
    if (previous_diff != 0)
    {// Can't fix the root node
      size_t parent_octree_index = previous_diff - 1;

      while (parent_octree_index > 0)
      {
        size_t parent_first_object_index = resulting_octree[parent_octree_index].start_index;
        octree_generation_position parent_position = sorted_positions[parent_first_object_index];
        size_t parent_level = get_level(parent_position);

        // Climb up the parent chain
        while (parent_octree_index > 0 && fix_level < parent_level &&
               resulting_octree[parent_octree_index - 1].start_index == parent_first_object_index)
        {
          --parent_octree_index;
          --parent_level;
        }

        if (parent_level == fix_level)
        {
          break;
        }

        --parent_octree_index;
      }

      // Write the children to it's position
      resulting_octree[parent_octree_index].children[
          get_level_position(current_position, fix_level + 1)
      ] = (resulting_octree + previous_diff);
    }
  }

  // Find the maximun and the minimum of the whole scene.
  // Scale the aabb to fit in the [0-1] space

  // For a fixed depth, let all objects place themself in the octree.
  // To do that, use 3bits per level to say where they are stored.

  // Sort the objects to group together objects at the same level.

  // Create the octree from that.
}