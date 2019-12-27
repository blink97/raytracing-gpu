#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "vector3.h"
#include "ray.h"
#include "colors.h"
#include "hit.h"
#include "light.h"
#include "thread_arg.h"

static struct color trace(struct scene scene, struct ray ray, float coef)
{
  if (coef < 0.01)
    return init_color(0, 0, 0);
  struct object obj;
  struct ray new_ray = collide(scene, ray, &obj);
  if (!vector3_is_zero(new_ray.direction))
  {
    struct color object = apply_light(scene, obj, new_ray);
    struct ray reflection_ray = ray_bounce(ray, new_ray);
    struct color reflection = trace(scene, reflection_ray, obj.nr * coef);
    struct color ret = color_add(reflection, color_mul(object, coef));
    return ret;
  }
  return init_color(0, 0, 0);
}

void raytrace(const scene &scene, struct color *output)
{
  struct vector3 u = vector3_normalize(scene.camera.u);
  struct vector3 v = vector3_normalize(scene.camera.v);
  struct vector3 w = vector3_cross(u, v);
  float L = scene.camera.width / (2 * tan(scene.camera.fov * M_PI / 360));
  struct vector3 C = vector3_add(scene.camera.position, vector3_scale(w, L));

  int halfw = scene.camera.width / 2;
  int halfh = scene.camera.height / 2;

  int startx = halfw;
  int stopx = -halfw;
  int starty = halfh;
  int stopy = -halfh;

  for (int j = startx; j > stopx; j--)
  {
    for (int i = starty; i > stopy; i--)
    {
      /* Aliasing here */
      struct color color = init_color(0, 0, 0);
      for (float k = i; k < i + 1; k += 0.5)
      {
        for (float l = j; l < j + 1; l += 0.5)
        {
          struct vector3 ui = vector3_scale(u, k);
          struct vector3 vj = vector3_scale(v, l);
          struct vector3 point = vector3_add(vector3_add(C, ui), vj);
          struct vector3 direction = vector3_normalize(vector3_sub(scene.camera.position, point));
          struct ray ray;
          ray.origin = point;
          ray.direction = direction;
          struct color tcolor = trace(scene, ray, 1);
          tcolor = color_mul(tcolor, 0.25);
          color = color_add(color, tcolor);
        }
      }
      output[(j + halfh) * scene.camera.width + (i + halfw)] = color;
    }
  }
}
