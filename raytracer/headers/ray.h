#ifndef RAY_H
# define RAY_H
# include "vector3.h"

struct ray {
  struct vector3 origin;
  struct vector3 direction;
};

struct ray init_ray(void);
struct ray ray_bounce(struct ray ray, struct ray normal);

#endif /* !RAY_H */
