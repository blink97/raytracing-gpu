#ifndef RAYTRACER_H
# define RAYTRACER_H

#include "scene.h"

void render(const scene &scene, char* buffer, int aliasing, std::ptrdiff_t stride, int pre_h, int pre_w);

#endif /* RAYTRACER_H */
