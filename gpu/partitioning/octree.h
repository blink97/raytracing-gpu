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

__global__ void find_scene_scale_basic(
  const struct AABB *const aabb,
  size_t nb_objects,
  struct AABB *resulting_scale
);

__global__ void find_scene_scale_shared(
  const struct AABB *const aabb,
  size_t nb_objects,
  struct AABB *resulting_scale
);

#endif /* !OCTREE_H */