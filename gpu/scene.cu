#include "scene.h"

# include <cstdlib>

/*
 * Add another object with the given triangle count to the scene.
 * The object is returned so that vertex and normal can be set.
 */
struct object *add_object_to_scene(struct scene *scene, uint32_t nb_triangles)
{
  scene->objects = (struct object *)realloc(scene->objects, sizeof(struct object) * (scene->object_count + 1));
  struct object *current_object = &scene->objects[scene->object_count++/* Increase the count of assigning */];

  // Create a default objects
  struct object new_object = {
    .vertex_and_normal = (vector3 *) malloc(sizeof(vector3) * 6/* 3 vertex and 3 normal */ * nb_triangles),
    .triangle_count = nb_triangles,
    .ka = { .x = 0, .y = 0, .z = 0 },
    .kd = { .x = 0, .y = 0, .z = 0 },
    .ks = { .x = 0, .y = 0, .z = 0 },
    .ns = 0,
    .ni = 1,
    .nr = 0,
    .d = 1
  };

  *current_object = new_object;
  return current_object;
}

/*
 * Get a new scene, containing only cuda memory so that
 * it can be used in a GPU context.
 */
struct scene to_cuda(const struct scene *const scene)
{
  struct scene cuda_scene = {
    .objects = nullptr,
    .object_count = scene->object_count,
    .lights = nullptr,
    .light_count = scene->light_count,
    .camera = scene->camera
  };

  cudaMalloc(&cuda_scene.objects, sizeof(struct object) * cuda_scene.object_count);
  cudaMalloc(&cuda_scene.lights, sizeof(struct light) * cuda_scene.light_count);

  cudaMemcpy(cuda_scene.objects, scene->objects, sizeof(struct object) * cuda_scene.object_count, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_scene.lights, scene->lights, sizeof(struct light) * cuda_scene.light_count, cudaMemcpyHostToDevice);

  // The object have been copied, but not all the vertex and normal.

  // TODO

  return cuda_scene;
}

vector3 *get_vertex(const struct object *const object, uint32_t triangle_index)
{
  return &object->vertex_and_normal[triangle_index * 6];
}

vector3 *get_normal(const struct object *const object, uint32_t triangle_index)
{
  return &object->vertex_and_normal[triangle_index * 6 + 3/* Skip the vertex part */];
}
