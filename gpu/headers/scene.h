#ifndef SCENE_H
# define SCENE_H

# include <stddef.h>
# include "vector3.h"

struct triangle {
  vector3 vertex[3];
  vector3 normal[3];
};

struct object {
  struct triangle *triangles;
  unsigned triangle_count;
  vector3 ka;// Ambient color
  vector3 kd;// Directional / point color
  vector3 ks;// Specular color
  float ns;// Specular light coefficient
  float ni;// Unused
  float nr;// Reflection coefficient
  float d;// Unused
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

struct scene {
  struct object *objects;
  size_t object_count;
  struct light *lights;
  size_t light_count;
  struct camera camera;
};

#endif /* !SCENE_H */
