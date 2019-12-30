#include <err.h>

#include "parser.h"
#include "partitioning/aabb.h"
#include "partitioning/octree.h"

#include <iostream>
#include <bitset>

#define TESTS_PATH "../../tests/"

#define CUBE TESTS_PATH "cube.svati"
#define ISLAND_SMOOTH TESTS_PATH "island_smooth.svati" // High objects count
#define DARK_NIGHT TESTS_PATH "dark-night.svati" // Second highest objects count
#define SPHERES TESTS_PATH "spheres.svati"

/*
 * File used for test purpose,
 * to see if the cuda functions have the correct output.
 */

void display_last_error()
{
  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
}


void display_GPU_memory()
{
  display_last_error();
  size_t free, total;
  cudaMemGetInfo(&free, &total);

  std::cout << "memory: " << free << "/" << total << std::endl;
}

void display_cuda_scene(const struct scene *cuda_scene)
{
  std::cout << "cuda scene:" << std::endl;
  struct scene CPU_scene;
  cudaMemcpy(&CPU_scene, cuda_scene, sizeof(struct scene), cudaMemcpyDefault);

  std::cout << "camera size: " << CPU_scene.camera.width << " " << CPU_scene.camera.height << std::endl;

  struct object *objects = new struct object[CPU_scene.object_count];
  cudaMemcpy(objects, CPU_scene.objects, sizeof(struct object) * CPU_scene.object_count, cudaMemcpyDefault);

  vector3 vertex[3];
  vector3 normal[3];

  for (size_t i = 0; i < CPU_scene.object_count; ++i)
  {
    std::cout << "new object with " << objects[i].triangle_count << " triangles" << std::endl;

    for (int j = 0; j < objects[i].triangle_count; ++j)
    {
      cudaMemcpy(vertex, get_vertex(objects[i].triangles, j), sizeof(vector3) * 3, cudaMemcpyDefault);
      cudaMemcpy(normal, get_normal(objects[i].triangles, j), sizeof(vector3) * 3, cudaMemcpyDefault);

      std::cout << "v : "
                << vertex[0].x << "," << vertex[0].y << "," << vertex[0].z << "\t| "
                << vertex[1].x << "," << vertex[1].y << "," << vertex[1].z << "\t| "
                << vertex[2].x << "," << vertex[2].y << "," << vertex[2].z << std::endl;
      std::cout << "vn: "
                << normal[0].x << "," << normal[0].y << "," << normal[0].z << "\t| "
                << normal[1].x << "," << normal[1].y << "," << normal[1].z << "\t| "
                << normal[2].x << "," << normal[2].y << "," << normal[2].z << std::endl;
    }
  }

  delete[] objects;
}

void display_aabbs(const struct AABB *aabbs, size_t nb_objects)
{
  std::cout << "displaying aabb" << std::endl;

  struct AABB *cpu_aabbs;
  cudaMallocHost(&cpu_aabbs, sizeof(struct AABB) * nb_objects);
  cudaMemcpy(cpu_aabbs, aabbs, sizeof(struct AABB) * nb_objects, cudaMemcpyDefault);

  std::cout << std::endl << nb_objects << " objects AABB (from: " << aabbs << " to: " << cpu_aabbs << ")" << std::endl;

  for (int i = 0; i < nb_objects; ++i)
  {//Display the aabb
    struct AABB current = cpu_aabbs[i];

    std::cout << current.min.x << "," << current.min.y << "," << current.min.z << " - "
              << current.max.x << "," << current.max.y << "," << current.max.z << std::endl;
  }

  cudaFreeHost(cpu_aabbs);
}

void display_positions(
  const octree_generation_position *positions,
  size_t *positions_sorted_index,
  size_t nb_objects)
{
  std::cout << "displaying positions" << (positions_sorted_index ? " sorted": " unsorted") << std::endl;

  octree_generation_position *cpu_positions;
  cudaMallocHost(&cpu_positions, sizeof(octree_generation_position) * nb_objects);
  cudaMemcpy(cpu_positions, positions, sizeof(octree_generation_position) * nb_objects, cudaMemcpyDefault);

  size_t *cpu_positions_sorted_index = nullptr;
  if (positions_sorted_index)
  {
    cudaMallocHost(&cpu_positions_sorted_index, sizeof(size_t) * nb_objects);
    cudaMemcpy(cpu_positions_sorted_index, positions_sorted_index, sizeof(size_t) * nb_objects, cudaMemcpyDefault);
  }

  for (int i = 0; i < nb_objects; ++i)
  {
    std::cout << "level: " << ((cpu_positions[i] & 0xFF000000) >> 24)
              << " " << std::bitset<24>(cpu_positions[i]);

    if (positions_sorted_index)
    {
      std::cout << " position: " << cpu_positions_sorted_index[i];
    }

    std::cout << std::endl;
  }

  cudaFreeHost(cpu_positions);
  if (positions_sorted_index) cudaFreeHost(cpu_positions_sorted_index);
}

void display_node_differences(
  const octree_generation_position *positions,
  size_t *node_differences,
  size_t nb_objects)
{
  std::cout << "displaying nodes differences" << std::endl;

  octree_generation_position *cpu_positions;
  cudaMallocHost(&cpu_positions, sizeof(octree_generation_position) * nb_objects);
  cudaMemcpy(cpu_positions, positions, sizeof(octree_generation_position) * nb_objects, cudaMemcpyDefault);

  size_t *cpu_node_differences;
  cudaMallocHost(&cpu_node_differences, sizeof(size_t) * nb_objects);
  cudaMemcpy(cpu_node_differences, node_differences, sizeof(size_t) * nb_objects, cudaMemcpyDefault);

  for (int i = 0; i < nb_objects; ++i)
  {
    std::cout << "level: " << ((cpu_positions[i] & 0xFF000000) >> 24)
              << " " << std::bitset<24>(cpu_positions[i])
              << " diff: " << cpu_node_differences[i] << std::endl;
  }

  cudaFreeHost(cpu_positions);
  cudaFreeHost(cpu_node_differences);
}

void test_partitioning(const struct scene *cuda_scene)
{
  display_cuda_scene(cuda_scene);

  struct scene CPU_scene;
  cudaMemcpy(&CPU_scene, cuda_scene, sizeof(struct scene), cudaMemcpyDefault);

  dim3 threadsPerBlock(32);
  dim3 numBlocks(ceil(CPU_scene.object_count * 1.0 / threadsPerBlock.x));

  std::cout << "kernel param: " << numBlocks.x << " " << threadsPerBlock.x << std::endl;
  std::cout << "nb_objects: " << CPU_scene.object_count << std::endl;

  // Compute the bounding box
  struct AABB *aabbs;
  cudaMalloc(&aabbs, sizeof(struct AABB) * CPU_scene.object_count);
  object_compute_bounding_box<<<numBlocks, threadsPerBlock>>>(cuda_scene, aabbs);
  display_aabbs(aabbs, CPU_scene.object_count);

  // Compute the global scale
  struct AABB *resulting_scale;
  cudaMalloc(&resulting_scale, sizeof(struct AABB));
  find_scene_scale_shared<<<numBlocks, threadsPerBlock>>>(aabbs, CPU_scene.object_count, resulting_scale);
  display_aabbs(resulting_scale, 1);

  // Compute the position of the objects
  octree_generation_position *positions;
  cudaMalloc(&positions, sizeof(octree_generation_position) * CPU_scene.object_count);
  position_object<<<numBlocks, threadsPerBlock>>>(aabbs, resulting_scale, positions, CPU_scene.object_count);
  display_positions(positions, nullptr, CPU_scene.object_count);

  // Sort the position of the objects
  size_t *positions_sorted_index;
  cudaMalloc(&positions_sorted_index, sizeof(size_t) * CPU_scene.object_count);
  single_thread_bubble_argsort<<<1, 1>>>(positions, positions_sorted_index, CPU_scene.object_count);
  display_positions(positions, positions_sorted_index, CPU_scene.object_count);

  // Get the number of nodes needed per each objects
  size_t *node_differences;
  cudaMalloc(&node_differences, sizeof(size_t) * CPU_scene.object_count);
  nodes_difference_array<<<numBlocks, threadsPerBlock>>>(positions, node_differences, CPU_scene.object_count);
  display_node_differences(positions, node_differences, CPU_scene.object_count);

  // Perform a prefix sum on it
  single_thread_nodes_difference_to_prefix_array<<<1, 1>>>(node_differences, CPU_scene.object_count);
  display_node_differences(positions, node_differences, CPU_scene.object_count);

  cudaFree(aabbs);
  cudaFree(resulting_scale);
  cudaFree(positions);
  cudaFree(node_differences);
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

  display_GPU_memory();

  //struct scene scene = parser(CUBE);
  //struct scene scene = parser(DARK_NIGHT);
  struct scene scene = parser(ISLAND_SMOOTH);
  //struct scene scene = parser(SPHERES);

  display_GPU_memory();

  struct scene *cuda_scene = to_cuda(&scene);

  test_partitioning(cuda_scene);
}
