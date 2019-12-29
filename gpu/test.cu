#include <err.h>

#include "parser.h"
#include "partitioning/aabb.h"
#include "partitioning/octree.h"

#include <iostream>

#define TESTS_PATH "../../tests/"

#define CUBE TESTS_PATH "cube.svati"
#define ISLAND_SMOOTH TESTS_PATH "island_smooth.svati" // High objects count
#define DARK_NIGHT TESTS_PATH "dark-night.svati" // Second highest objects count
#define SPHERES TESTS_PATH "spheres.svati"

/*
 * File used for test purpose,
 * to see if the cuda functions have the correct output.
 */

void display_aabbs(struct AABB *aabbs, size_t nb_objects)
{
  struct AABB *cpu_aabbs = new struct AABB[nb_objects];
  cudaMemcpy(&cpu_aabbs, aabbs, sizeof(struct AABB) * nb_objects, cudaMemcpyDefault);
  std::cout << std::endl << "objects AABB" << std::endl;

  for (int i = 0; i < nb_objects; ++i)
  {//Display the aabb
    std::cout << cpu_aabbs[i].min.x << "," << cpu_aabbs[i].min.y << "," << cpu_aabbs[i].min.z << " - "
              << cpu_aabbs[i].max.x << "," << cpu_aabbs[i].max.y << "," << cpu_aabbs[i].max.z << std::endl;
  }

  delete[] cpu_aabbs;
}

void test_partitioning(struct scene *cuda_scene)
{
  dim3 threadsPerBlock(32);
  dim3 numBlocks(ceil(cuda_scene->object_count * 1.0 / threadsPerBlock.x));

  // Compute the bounding box
  struct AABB *aabbs;
  cudaMalloc(&aabbs, sizeof(struct AABB) * cuda_scene->object_count);
  object_compute_bounding_box<<<numBlocks, threadsPerBlock>>>(cuda_scene, aabbs);

  display_aabbs(aabbs, cuda_scene->object_count);

  // Compute the global scale
  struct AABB resulting_scale;
  find_scene_scale_shared<<<numBlocks, threadsPerBlock>>>(aabbs, cuda_scene->object_count, &resulting_scale);

  display_aabbs(&resulting_scale, 1);

  octree_generation_position *positions;
  cudaMalloc(&positions, sizeof(octree_generation_position) * cuda_scene->object_count);
  position_object<<<numBlocks, threadsPerBlock>>>(aabbs, &resulting_scale, positions, cuda_scene->object_count);

  cudaFree(aabbs);
  cudaFree(positions);
}

int main(int argc, char *argv[])
{

#  if defined(LAYOUT_FRAGMENTED)
  std::cout << "Using fragmented layout" << std::endl;
#  elif defined(LAYOUT_AOS)
  std::cout << "Using array of structures (AOS) layout" << std::endl;
#  else /* LAYOUT_SOA */
  std::cout << "Using structure of arrays (SOA) layout" << std::endl;
#  endif

  struct scene scene = parser(CUBE);
  struct scene cuda_scene = to_cuda(&scene);

  test_partitioning(&cuda_scene);
}
