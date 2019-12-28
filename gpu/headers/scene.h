#ifndef SCENE_H
# define SCENE_H

# include <stddef.h>
# include <cstdint>

# include "vector3.h"

enum light_type {
  AMBIENT,
  DIRECTIONAL,
  POINT,
  SPECULAR
};

struct light {
  enum light_type type;
  float r;
  float g;
  float b;
  vector3 v;

};

struct camera {
  int width;
  int height;
  vector3 position;
  vector3 u;
  vector3 v;
  float fov;
};

struct object;
struct scene;

/*
 * Get an empty scene that can be used whatever is the memory layout used.
 */
struct scene empty_scene();

/*
 * Add another object with the given triangle count to the scene.
 * The object is returned so that vertex and normal can be set.
 */
struct object *add_object_to_scene(struct scene *scene, uint32_t nb_triangles);

/*
 * Get a new scene, containing only cuda memory so that
 * it can be used in a GPU context.
 */
struct scene to_cuda(const struct scene *const scene);

/*
 * The way to get vertex and normal depends on how
 * the layout is defined in memory, so functions need to be used
 * to allows simple change.
 * Those functions can be used both by the host and device code.
 */
__host__ __device__ vector3 *get_vertex(const struct triangles_layout triangles, uint32_t triangle_index);
__host__ __device__ vector3 *get_normal(const struct triangles_layout triangles, uint32_t triangle_index);

/* Layout dependent code */
# if defined(LAYOUT_FRAGMENTED)

struct triangles_layout {
  /*
   * Contains vertex[3], followed by normal[3], repeated triangle_count times.
   * Each data pointer is a new allocation.
   */
  vector3 *data;
};

struct scene_objects_additional_data {
  /* No more additionnal data */
};

# elif defined(LAYOUT_AOS)

struct triangles_layout {
  /*
   * Contains vertex[3], followed by normal[3], repeated triangle_count times.
   * Each data pointer is a not an allocation,
   * but a pointer of allocated memory in the scene
   * (reduce global fragmentation).
   */
  vector3 *data;
};

struct scene_objects_additional_data {
  // The global array of vertex and normal
  // to reduce memory fragmentation
  vector3 *vertex_and_normal;
};

# else /* LAYOUT_SOA */

struct triangles_layout {
  /*
   * Contains vertex[3] and normal[3] separately, each repeated triangle_count times.
   * Each data pointer is a not an allocation,
   * but a pointer of allocated memory in the scene
   * (reduce global fragmentation).
   */
  vector3 *vertex;
  vector3 *normal;
};

struct scene_objects_additional_data {
  // The allocated arrays of normal and vertex that
  // each triangles reference.
  vector3 *vertex;
  vector3 *normal;
};

# endif
/* End of layout dependent code */

struct object {
  /*
   * Fields of this structures must not be accessed directly,
   * as they can change with the layout used.
   */
  struct triangles_layout triangles;

  uint32_t triangle_count;
  vector3 ka;// Ambient color
  vector3 kd;// Directional / point color
  vector3 ks;// Specular color
  float ns;// Specular light coefficient
  float ni;// Unused
  float nr;// Reflection coefficient
  float d;// Unused
};

struct scene {
  /*
   * Fields of this structure must not be directly
   * accessed as they can change based on the layout.
   */
  struct scene_objects_additional_data objects_data;
  struct object *objects;
  size_t object_count;
  struct light *lights;
  size_t light_count;
  struct camera camera;
  // Have the global triangle count pre-computed to prevents
  // recomputing it each times.
  size_t triangle_count;
};

#endif /* !SCENE_H */
