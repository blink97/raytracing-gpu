#include "scene.h"

# include <cstdlib>


/* Layout dependent code */
# if defined(LAYOUT_FRAGMENTED)

__host__ __device__ vector3 *get_vertex(const struct triangles_layout triangles, uint32_t triangle_index)
{
  return &triangles.data[triangle_index * 6];
}

__host__ __device__ vector3 *get_normal(const struct triangles_layout triangles, uint32_t triangle_index)
{
  return &triangles.data[triangle_index * 6 + 3/* Skip the vertex part */];
}


# elif defined(LAYOUT_AOS)


__host__ __device__ vector3 *get_vertex(const struct triangles_layout triangles, uint32_t triangle_index)
{
  return &triangles.data[triangle_index * 6];
}

__host__ __device__ vector3 *get_normal(const struct triangles_layout triangles, uint32_t triangle_index)
{
  return &triangles.data[triangle_index * 6 + 3/* Skip the vertex part */];
}


# else /* LAYOUT_SOA */


__host__ __device__ vector3 *get_vertex(const struct triangles_layout triangles, uint32_t triangle_index)
{
  return &triangles.vertex[triangle_index * 3];
}

__host__ __device__ vector3 *get_normal(const struct triangles_layout triangles, uint32_t triangle_index)
{
  return &triangles.normal[triangle_index * 3];
}


# endif
/* End of layout dependent code */

/**
 * Rewrite the triangles pointers so that they point to the correct position
 */
static void rewrite_pointers(const struct scene *scene)
{
  size_t offset = 0;
  for (uint32_t i = 0; i < scene->object_count; ++i)
  {
    // Get the object back on the CPU
    struct object current_object;

    cudaMemcpy(
      &current_object,
      scene->objects + i,
      sizeof(struct object),
      cudaMemcpyDefault
    );

      /* Layout dependent code */
#  if defined(LAYOUT_FRAGMENTED)

    /* Nothing to do here */

#  elif defined(LAYOUT_AOS)

    current_object.triangles.data = scene->objects_data.vertex_and_normal + offset;

    offset += 6 * current_object.triangle_count;

#  else /* LAYOUT_SOA */

    current_object.triangles.vertex = scene->objects_data.vertex + offset;
    current_object.triangles.normal = scene->objects_data.normal + offset;

    offset += 3 * current_object.triangle_count;

#  endif
  /* End of layout dependent code */

    // Replace the object at it's current location
    cudaMemcpy(
      scene->objects + i,
      &current_object,
      sizeof(struct object),
      cudaMemcpyDefault
    );
  }
}


struct scene empty_scene()
{
  struct scene scene = {
    .objects_data = {/* Default initialisation */},
    .objects = nullptr,
    .object_count = 0,
    .lights = nullptr,
    .light_count = 0,
    .camera = {/* Use default init */},
    .triangle_count = 0,
  };

  return scene;
}


/*
 * Add another object with the given triangle count to the scene.
 * The object is returned so that vertex and normal can be set.
 */
struct object *add_object_to_scene(struct scene *scene, uint32_t nb_triangles)
{
  scene->objects = (struct object *)realloc(scene->objects, sizeof(struct object) * (scene->object_count + 1));

  /* Layout dependent code */
#  if defined(LAYOUT_FRAGMENTED)

  struct triangles_layout triangles = {
    .data = (vector3 *) malloc(sizeof(vector3) * 6/* 3 vertex and 3 normal */ * nb_triangles)
  };

#  elif defined(LAYOUT_AOS)

  // Extend the global triangles storage
  vector3 *old_ptr = scene->objects_data.vertex_and_normal;

  scene->objects_data.vertex_and_normal = (vector3 *)realloc(
    scene->objects_data.vertex_and_normal,
    sizeof(vector3) * 6/* 3 vertex and 3 normal */ * (scene->triangle_count + nb_triangles)
  );

  if (old_ptr != scene->objects_data.vertex_and_normal)
  {// Rewrite the pointers as it may have changed.
    rewrite_pointers(scene);
  }

  struct triangles_layout triangles = {
    .data = &scene->objects_data.vertex_and_normal[scene->triangle_count * 6]
  };

#  else /* LAYOUT_SOA */

  vector3 *old_vertex_ptr = scene->objects_data.vertex;
  vector3 *old_normal_ptr = scene->objects_data.normal;

  // Extend the global vertex and normal storage
  scene->objects_data.vertex = (vector3 *)realloc(scene->objects_data.vertex, sizeof(vector3) * 3 * (scene->triangle_count + nb_triangles));
  scene->objects_data.normal = (vector3 *)realloc(scene->objects_data.normal, sizeof(vector3) * 3 * (scene->triangle_count + nb_triangles));

  if (old_vertex_ptr != scene->objects_data.vertex || old_normal_ptr != scene->objects_data.normal)
  {// Rewrite the pointers as it may have changed.
    rewrite_pointers(scene);
  }

  struct triangles_layout triangles = {
    .vertex = &scene->objects_data.vertex[scene->triangle_count * 3],
    .normal = &scene->objects_data.normal[scene->triangle_count * 3]
  };

#  endif
  /* End of layout dependent code */

  // Create a default objects
  struct object new_object = {
    .triangles = triangles,
    .triangle_count = nb_triangles,
    .ka = { .x = 0, .y = 0, .z = 0 },
    .kd = { .x = 0, .y = 0, .z = 0 },
    .ks = { .x = 0, .y = 0, .z = 0 },
    .ns = 0,
    .ni = 1,
    .nr = 0,
    .d = 1
  };

  struct object *current_object = &scene->objects[scene->object_count];
  *current_object = new_object;

  scene->object_count++;
  scene->triangle_count += nb_triangles;

  return current_object;
}

/*
 * Get a new scene, containing only cuda memory so that
 * it can be used in a GPU context.
 */
struct scene to_cuda(const struct scene *const scene)
{
  struct scene cuda_scene = {
    .objects_data = scene->objects_data,
    .objects = nullptr,
    .object_count = scene->object_count,
    .lights = nullptr,
    .light_count = scene->light_count,
    .camera = scene->camera,
    .triangle_count = scene->triangle_count,
  };

  cudaMalloc(&cuda_scene.objects, sizeof(struct object) * cuda_scene.object_count);
  cudaMalloc(&cuda_scene.lights, sizeof(struct light) * cuda_scene.light_count);

  cudaMemcpy(cuda_scene.objects, scene->objects, sizeof(struct object) * cuda_scene.object_count, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_scene.lights, scene->lights, sizeof(struct light) * cuda_scene.light_count, cudaMemcpyHostToDevice);

  /* Layout dependent code */
#  if defined(LAYOUT_FRAGMENTED)

  // Copy all vertex and normal arrays in each objects.
  for (uint32_t i = 0; i < scene->object_count; ++i)
  {
    size_t mem_size = sizeof(vector3) * 6 /* 3 vertex and 3 normal */ * scene->objects[i].triangle_count;

    struct object current_object = scene->objects[i];
    cudaMalloc(&current_object.triangles.data, mem_size);

    // Copy the triangle
    cudaMemcpy(
      current_object.triangles.data,
      scene->objects[i].triangles.data,
      mem_size,
      cudaMemcpyHostToDevice
    );

    // Copy the object back to GPU
    cudaMemcpy(
      &cuda_scene.objects[i],
      &current_object,
      sizeof(struct object),
      cudaMemcpyHostToDevice
    );
  }

#  elif defined(LAYOUT_AOS)

  // Copy the global vertex and normal array,
  // and rewrite the objects triangles pointers.

  size_t mem_size = sizeof(vector3) * 6 * scene->triangle_count;

  cudaMalloc(&cuda_scene.objects_data.vertex_and_normal, mem_size);
  cudaMemcpy(
    cuda_scene.objects_data.vertex_and_normal,
    scene->objects_data.vertex_and_normal,
    mem_size,
    cudaMemcpyHostToDevice
  );

  // Rewrite the pointers so that they point to the correct value.
  rewrite_pointers(&cuda_scene);

#  else /* LAYOUT_SOA */

  // Copy the globals vertex and normal arrays,
  // and rewrite the objects triangles pointers.

  size_t mem_size = sizeof(vector3) * 3 * scene->triangle_count;

  cudaMalloc(&cuda_scene.objects_data.vertex, mem_size);
  cudaMalloc(&cuda_scene.objects_data.normal, mem_size);
  cudaMemcpy(
    cuda_scene.objects_data.vertex,
    scene->objects_data.vertex,
    mem_size,
    cudaMemcpyHostToDevice
  );
  cudaMemcpy(
    cuda_scene.objects_data.normal,
    scene->objects_data.normal,
    mem_size,
    cudaMemcpyHostToDevice
  );

  // Rewrite the pointers so that they point to the correct value.
  rewrite_pointers(&cuda_scene);

#  endif

  return cuda_scene;
}
