#ifndef HIT_H
# define HIT_H

# include "scene.h"
# include "ray.h"

__host__ __device__ struct ray collide(struct scene scene, struct ray ray, struct object *hit);
__host__ __device__ float collide_dist(struct scene scene, struct ray ray);

#endif /* !HIT_H */
