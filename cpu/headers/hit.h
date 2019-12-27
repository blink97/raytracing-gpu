#ifndef HIT_H
# define HIT_H

# include "scene.h"
# include "ray.h"

struct ray collide(struct scene scene, struct ray ray, struct object *hit);
float collide_dist(struct scene scene, struct ray ray);

#endif /* !HIT_H */
