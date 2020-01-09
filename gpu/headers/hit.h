#ifndef HIT_H
# define HIT_H

# include "scene.h"
# include "ray.h"

__device__ struct ray collide(struct scene* scene, struct object* objects, struct ray ray, struct object *hit);
__device__ float collide_dist(struct scene* scene, struct object* objects, struct ray ray);

#endif /* !HIT_H */
