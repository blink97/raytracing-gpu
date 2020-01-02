#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "light.h"
#include "hit.h"

__device__ static float cufmax(float a, float b) {
	return a > b ? a : b;
}

__device__ static void apply_specular(struct color *color, struct ray incident,
                           struct ray normal, struct object obj)
{
  struct color kcolor = init_color(obj.ks.x, obj.ks.y, obj.ks.z);
  vector3 V = vector3_sub(incident.origin, normal.origin);
  vector3 R =
    vector3_sub(incident.direction,
                vector3_scale(normal.direction,
                              2 * vector3_dot(normal.direction,
                                              incident.direction)));
  R = vector3_normalize(R);
  V = vector3_normalize(V);
  float Ls = pow(cufmax(vector3_dot(R, V), 0.0), obj.ns);
  kcolor = color_mul(&kcolor, Ls);
  *color = color_add(color, &kcolor);
}

__device__ static int has_direct_hit(struct scene* scene, struct ray light_ray)
{
  float fdist = collide_dist(scene, light_ray);
  if (fdist < 1)
  if (fdist == 0)
    return 0;
  return 1;
}


//__device__ struct color operator*(struct color* a, struct color* b) {
//	return init_color((float(a.r) / 255.0f) * (float(b.r) / 255.0f),
//	                    (float(a.g) / 255.0f) * (float(b.g) / 255.0f),
//	                    (float(a.b) / 255.0f) * (float(b.b) / 255.0f));
//}

//__host__ __device__ struct color operator*(struct color &a, const float coef) {
//	return init_color(float(a.r) / 255 * coef, float(a.g) / 255 * coef, float(a.b) / 255 * coef);
//}


__device__ struct color apply_light(struct scene* scene, struct object* object,
                         struct ray point)
{
  struct color color = init_color(0, 0, 0);
   for (size_t i = 0; i < scene->light_count; i++)
  {
    struct light light = scene->lights[i];
    switch (light.type)
    {
      case AMBIENT:
        {
        	struct color l_color = init_color(light.r, light.g, light.b);
        	struct color obj_color = init_color(object->ka.x, object->ka.y, object->ka.z);
          struct color tmp = color_mults(l_color,obj_color);

          color = color_add(&color, &tmp);
          break;
        }
      case DIRECTIONAL:
        {
          struct ray light_ray;
          light_ray.origin = point.origin;
          light_ray.direction = vector3_scale(light.v, -1);

          if (!has_direct_hit(scene, light_ray))
          {
            vector3 L = vector3_scale(light.v, -1);
            vector3 N = point.direction;

            struct color l_color = init_color(light.r, light.g, light.b);
            struct color obj_color = init_color(object->kd.x, object->kd.y, object->kd.z);
            struct color tmp = color_mults(l_color, obj_color);


            tmp = color_mul(&tmp, vector3_dot(L, N));

            light_ray.direction = light.v;
            light_ray.origin = vector3_add(light_ray.origin,
                                           vector3_scale(light_ray.direction,
                                                         -10));
            apply_specular(&tmp, light_ray, point, *object);

            color = color_add(&color, &tmp);
          }
          break;
        }
      case POINT:
        {
          vector3 L = vector3_scale(light.v, -1);
          vector3 N = point.direction;
          if (vector3_dot(L, N) < 0)
            N = vector3_scale(N, -1);
          struct ray light_ray;
          light_ray.origin = point.origin;
          light_ray.direction = vector3_sub(light.v, point.origin);
          float dist = vector3_length(vector3_sub(light.v, point.origin));
          if (!has_direct_hit(scene, light_ray))
          {
        	struct color l_color = init_color(light.r, light.g, light.b);
        	struct color obj_color = init_color(object->kd.x, object->kd.y, object->kd.z);
            struct color tmp = color_mults(l_color, obj_color);
            tmp = color_mul(&tmp, vector3_dot(L, N) * 1 / dist);
            light_ray.direction = vector3_sub(light.v, point.origin);
            light_ray.origin = vector3_add(light_ray.origin,
                                           vector3_scale(light_ray.direction,
                                                         -10));
            apply_specular(&tmp, light_ray, point, *object);
            color = color_add(&color, &tmp);
          }
        break;
        }
      default:
        continue;
        break;
    }
  }
  return color;
}
