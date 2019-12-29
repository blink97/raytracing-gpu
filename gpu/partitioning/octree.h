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
  // The objects index in the scene at which this level end.
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
 * All functions exported under this are only present
 * so that they can be benchmarked together.
 */

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

#endif /* !OCTREE_H */