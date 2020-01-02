#include "ray.h"
#include "vector3.h"

__device__ struct ray init_ray(void)
{
  struct ray ret;
  ret.origin.x = 0;
  ret.origin.y = 0;
  ret.origin.z = 0;
  ret.direction.x = 0;
  ret.direction.y = 0;
  ret.direction.z = 0;
  return ret;
}

__device__ struct ray ray_bounce(struct ray* ray, struct ray* normal)
{
  struct ray ret;
  ret.origin = normal->origin;
  ret.direction = vector3_sub(ray->direction,
                              vector3_scale(normal->direction,
                                           2 * vector3_dot(normal->direction,
                                                           ray->direction)));
  return ret;
}
