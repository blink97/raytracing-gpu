#ifndef SCENE_H
# define SCENE_H

# include <stddef.h>
# include "vector3.h"

struct triangle {
  struct vector3 vertex[3];
  struct vector3 normal[3];
};

struct object {
  struct triangle *triangles;
  unsigned triangle_count;
  struct vector3 ka;
  struct vector3 kd;
  struct vector3 ks;
  float ns;
  float ni;
  float nr;
  float d;
};

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
  struct vector3 v;

};

struct camera {
  int width;
  int height;
  struct vector3 position;
  struct vector3 u;
  struct vector3 v;
  float fov;
};

struct scene {
  struct object *objects;
  size_t object_count;
  struct light *lights;
  size_t light_count;
  struct camera camera;
};

#endif /* !SCENE_H */
