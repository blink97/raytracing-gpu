#ifndef RAYTRACER_H
# define RAYTRACER_H

#include "scene.h"

void raytrace(const scene &scene, struct color *output, int aliasing = 1 /* 2 for x4, 4 for x16... */);

#endif /* RAYTRACER_H */
