#ifndef RAYTRACER_H
# define RAYTRACER_H

#include "scene.h"

void render(const scene &scene, char* buffer, int aliasing, std::ptrdiff_t stride);

#endif /* RAYTRACER_H */
