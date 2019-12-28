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
__host__ __device__ vector3 *get_vertex(const struct object *const object, uint32_t triangle_index);
__host__ __device__ vector3 *get_normal(const struct object *const object, uint32_t triangle_index);

/* Layout dependent code */
# if defined(LAYOUT_FRAGMENTED)

# elif defined(LAYOUT_AOS)

# else /* LAYOUT_SOA */

# endif
/* End of layout dependent code */

struct object {
  // Contains vertex[3], directly followed by normal[3], repeated triangle_count times.
  vector3 *vertex_and_normal;
  unsigned triangle_count;
  vector3 ka;// Ambient color
  vector3 kd;// Directional / point color
  vector3 ks;// Specular color
  float ns;// Specular light coefficient
  float ni;// Unused
  float nr;// Reflection coefficient
  float d;// Unused
};

struct scene {
  struct object *objects;
  size_t object_count;
  struct light *lights;
  size_t light_count;
  struct camera camera;
};

#endif /* !SCENE_H */
