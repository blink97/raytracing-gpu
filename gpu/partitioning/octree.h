#ifndef OCTREE_H
# define OCTREE_H

#include "scene.h"
#include "aabb.h"

/*
 * The octree will reorder objects in the scene,
 * making objects present at the top level always
 * be the first checked, while the most specific will
 * be placed more at the end of the list.
 */
struct octree
{
  // The current center of the octree.
  // All children are placed around this center.
  vector3 center;

  // The objects index in the scene at which this level start.
  size_t start_index;
  // The objects index in the scene at which this level end, excluded.
  // Excluding the value allows for simple iteration: while (i < end_index),
  // and at the same times, handling of nodes only
  // containing children and no objects.
  size_t end_index;

  // All the children of the octree.
  // Can be nullptr if no child is present.
  // One children can be present, while the other is absent.
  struct octree *children[8];
};

/**
 * Create an octree from the given scene.
 * The scene will be modified (objects order will be changed).
 * The resulting octree will be stored in the pointer given as parameter.
 */
void create_octree(
  struct scene *scene,
  struct AABB *aabb,
  struct octree **octree
);

/*
 * Contains the position of an object into a octree.
 * This value contains two information:
 *  - the level depth
 *  - the position to consider
 * They are layed like this: [depth 31-28][position 27-0]
 * Laying them like this allows to sort them the wanted way,
 * so that the range can be contructed without any problems.
 */
typedef uint32_t octree_generation_position;

/*
 * Get the level associated with the given position
 */
__host__ __device__ uint8_t get_level(octree_generation_position position);

/*
 * Get the position in the octree node at the given level.
 */
__host__ __device__ uint8_t get_level_position(octree_generation_position position, uint8_t level);

/*
 * Find the scale of the scene, so that the rest
 * of the algorithm can be done in the [0-1] cube,
 * and rescale the octree back on creation.
 */
__global__ void find_scene_scale_basic(
  const struct AABB *const aabbs,
  size_t nb_objects,
  struct AABB *resulting_scale
);

__global__ void find_scene_scale_shared(
  const struct AABB *const aabbs,
  size_t nb_objects,
  struct AABB *resulting_scale
);

/*
 * Get the position in the octree of all aabb,
 * so that they can be stored without any problem.
 * This only return the expected position of the object,
 * without taking into account some optimisation,
 * as going in the upper node if there is few vertex in it.
 */
__global__ void position_object(
  const struct AABB *const aabbs,
  const struct AABB *const scale,
  octree_generation_position *positions,
  size_t nb_objects
);


/*
 * Sort the positions so that the tree creation can be done
 * simply, as objects on the same octree are placed next to
 * each other.
 */
__global__ void single_thread_bubble_argsort(
  octree_generation_position *positions,
  size_t *indexes,
  size_t nb_objects
);

/*
 * Get the number of nodes in the octree that needs to be created
 * for each position, excluding the ones already created
 * by other nodes. Positions must be sorted, as it allows
 * to easily get this number for each position, only having to
 * look at the previous element in the array to get the difference
 * in the number of octree nodes.
 */
__global__ void nodes_difference_array(
  const octree_generation_position *const sorted_positions,
  size_t *nodes_difference,
  size_t nb_objects
);


/*
 * Compute the prefix sum array for the nodes difference.
 * This allows to simply knows how many octree nodes must be
 * created in advance, and allowing fast octree construction,
 * as the octree can be fully constructed in parallel.
 * The number of octree nodes that need to be created
 * is the last value in the array.
 */
__global__ void single_thread_nodes_difference_to_prefix_array(
  size_t *nodes_difference,
  size_t nb_objects
);

/*
 * Create the full octree.
 * The number of octree node created is the last value
 * of the prefix array.
 */
__global__ void create_octree(
  const octree_generation_position *const sorted_positions,
  const size_t *const nodes_difference,
  size_t nb_objects,
  const struct AABB *const scale,
  struct octree *resulting_octree
);

#endif /* !OCTREE_H */