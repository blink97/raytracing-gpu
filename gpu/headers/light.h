#ifndef LIGHT_H
# define LIGHT_H
# include "colors.h"
# include "scene.h"
# include "ray.h"

struct color apply_light(struct scene, struct object, struct ray);

#endif /* !LIGHT_H */
