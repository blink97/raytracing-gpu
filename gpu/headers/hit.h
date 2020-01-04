#ifndef HIT_H
# define HIT_H

# include "scene.h"
# include "ray.h"

__device__ struct ray collide(const struct scene* scene, struct ray ray, struct object *hit);
__device__ float collide_dist(const struct scene* scene, struct ray ray);

#endif /* !HIT_H */
