#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif
#include <math.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#include "printer.h"
#include "scene.h"
#include "vector3.h"
#include "parser.h"
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

void * raytrace_thread(void *args)
{
  struct thread_arg *arg = args;
  int startx = arg->startx;
  int stopx = arg->stopx;
  int starty = arg->starty;
  int stopy = arg->stopy;
  int halfh = arg->halfh;
  int halfw = arg->halfw;
  struct scene scene = arg->scene;
  struct color *output_tab = arg->output_tab;
  struct vector3 v = arg->v;
  struct vector3 u = arg->u;
  struct vector3 C = arg->C;
  for (int j = startx; j > stopx; j--)
  {
    for (int i = starty; i > stopy; i--)
    {
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
      output_tab[(j + halfh) * scene.camera.width + (i + halfw)] = color;
    }
  }
  int *i = malloc(4);
  *i = 100;
  pthread_exit((void *)i);
}

void raytrace(const char *input, const char *output)
{
  struct scene scene = parser(input);
  struct vector3 u = vector3_normalize(scene.camera.u);
  struct vector3 v = vector3_normalize(scene.camera.v);
  struct vector3 w = vector3_cross(u, v);
  float L = scene.camera.width / (2 * tan(scene.camera.fov * M_PI / 360));
  struct vector3 C = vector3_add(scene.camera.position, vector3_scale(w, L));
  FILE *out = open_output(output, scene.camera.width, scene.camera.height);

  int halfw = scene.camera.width / 2;
  int halfh = scene.camera.height / 2;
  struct color output_tab[(scene.camera.width + 1) * (scene.camera.height + 1)];
  pthread_t tid[4];
  struct thread_arg args[4];
  for (int i = 0; i < 4; i++)
  {
    args[i] =
      init_thread_arg(output_tab, v, u, C);
    args[i].scene = scene;
  }
  args[0].startx = halfh;
  args[0].starty = halfw;
  args[0].stopx = 0;
  args[0].stopy = 0;
  args[1].startx = halfh;
  args[1].starty = 0;
  args[1].stopx = 0;
  args[1].stopy = -halfw;
  args[2].startx = 0;
  args[2].starty = halfw;
  args[2].stopx = -halfh;
  args[2].stopy = 0;
  args[3].startx = 0;
  args[3].starty = 0;
  args[3].stopx = -halfh;
  args[3].stopy = -halfw;

  for (int i = 0; i < 4; i++)
  {
    args[i].halfh = halfh;
    args[i].halfw = halfw;
    pthread_create(&tid[i], NULL, raytrace_thread, args + i);
  }

  for (int i = 0; i < 4; i++)
  {
    pthread_join(tid[i], NULL);
  }
  for (int j = scene.camera.height; j > 0; j--)
  {
    for (int i = scene.camera.width; i > 0; i--)
    {
      print_color(output_tab[j * scene.camera.width + i], out);
    }
  }
  return;
}
