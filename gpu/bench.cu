#include <benchmark/benchmark.h>
#include <functional>

#include "parser.h"
#include "partitioning/aabb.h"
#include "partitioning/octree.h"
#include "partitioning/prefix_sum.h"
#include "partitioning/sort.h"

#define TESTS_PATH "../../tests/"

// All tests
#define CUBE TESTS_PATH "cube.svati"
#define ISLAND_SMOOTH TESTS_PATH "island_smooth.svati" // High objects count
#define DARK_NIGHT TESTS_PATH "dark-night.svati" // Second highest objects count
#define SPHERES TESTS_PATH "spheres.svati"

#define FULL_BENCHMARK(function) \
  BENCHMARK_CAPTURE(function, simple_cube, CUBE); \
  BENCHMARK_CAPTURE(function, island_smooth, ISLAND_SMOOTH); \
  BENCHMARK_CAPTURE(function, dark_night, DARK_NIGHT); \
  BENCHMARK_CAPTURE(function, spheres, SPHERES);

/*
 * Benchmark the parser to see memory alignement difference
 */
void BM_parser(benchmark::State& st, const char *filename)
{
  for (auto _ : st)
    parser(filename);
}

FULL_BENCHMARK(BM_parser);


/*
 * Benchmark the AABB creation with the object strategy
 */
void BM_aabb_object(benchmark::State& st, const char *filename)
{
  struct scene scene = parser(filename);
  struct scene *cuda_scene = to_cuda(&scene);

  struct AABB *aabbs;
  cudaMalloc(&aabbs, sizeof(struct AABB) * scene.object_count);

  // Compute the boundign box per objects.
  dim3 threadsPerBlock(32);
  dim3 numBlocks(ceil(scene.object_count * 1.0 / threadsPerBlock.x));

  for (auto _ : st)
    object_compute_bounding_box<<<numBlocks, threadsPerBlock>>>(cuda_scene, aabbs);

  cudaFree(aabbs);
}

//FULL_BENCHMARK(BM_aabb_object);


/*
 * Benchmark the scene scale finding with the basic strategy.
 */
void BM_find_scene_scale_basic(benchmark::State& st, const char *filename)
{
  struct scene scene = parser(filename);
  struct scene *cuda_scene = to_cuda(&scene);

  dim3 threadsPerBlock(32);
  dim3 numBlocks(ceil(scene.object_count * 1.0 / threadsPerBlock.x));

  // Compute the bounding box
  struct AABB *aabbs;
  cudaMalloc(&aabbs, sizeof(struct AABB) * scene.object_count);
  object_compute_bounding_box<<<numBlocks, threadsPerBlock>>>(cuda_scene, aabbs);

  // Compute the global scale
  struct AABB *resulting_scale;
  cudaMalloc(&resulting_scale, sizeof(struct AABB));

  for (auto _ : st)
    find_scene_scale_basic<<<numBlocks, threadsPerBlock>>>(aabbs, scene.object_count, resulting_scale);

  cudaFree(aabbs);
}

FULL_BENCHMARK(BM_find_scene_scale_basic);


/*
 * Benchmark the scene scale finding with  the shared strategy.
 */
void BM_find_scene_scale_shared(benchmark::State& st, const char *filename)
{
  struct scene scene = parser(filename);
  struct scene *cuda_scene = to_cuda(&scene);

  dim3 threadsPerBlock(32);
  dim3 numBlocks(ceil(scene.object_count * 1.0 / threadsPerBlock.x));

  // Compute the bounding box
  struct AABB *aabbs;
  cudaMalloc(&aabbs, sizeof(struct AABB) * scene.object_count);
  object_compute_bounding_box<<<numBlocks, threadsPerBlock>>>(cuda_scene, aabbs);

  // Compute the global scale
  struct AABB *resulting_scale;
  cudaMalloc(&resulting_scale, sizeof(struct AABB));

  for (auto _ : st)
    find_scene_scale_shared<<<numBlocks, threadsPerBlock>>>(aabbs, scene.object_count, resulting_scale);

  cudaFree(aabbs);
}

FULL_BENCHMARK(BM_find_scene_scale_shared);

/*
 * Benchmark the octree position creation.
 */
void BM_position_object(benchmark::State& st, const char *filename)
{
  struct scene scene = parser(filename);
  struct scene *cuda_scene = to_cuda(&scene);

  dim3 threadsPerBlock(32);
  dim3 numBlocks(ceil(scene.object_count * 1.0 / threadsPerBlock.x));

  // Compute the bounding box
  struct AABB *aabbs;
  cudaMalloc(&aabbs, sizeof(struct AABB) * scene.object_count);
  object_compute_bounding_box<<<numBlocks, threadsPerBlock>>>(cuda_scene, aabbs);

  // Compute the global scale
  struct AABB *resulting_scale;
  cudaMalloc(&resulting_scale, sizeof(struct AABB));

  find_scene_scale_shared<<<numBlocks, threadsPerBlock>>>(aabbs, scene.object_count, resulting_scale);

  octree_generation_position *positions;
  cudaMalloc(&positions, sizeof(octree_generation_position) * scene.object_count);

  for (auto _ : st)
    position_object<<<numBlocks, threadsPerBlock>>>(aabbs, resulting_scale, positions, scene.object_count);

  cudaFree(aabbs);
  cudaFree(positions);
}

FULL_BENCHMARK(BM_position_object);


/*
 * Benchmark the octree position sorting.
 */
void BM_sort_object_bubble_sort(benchmark::State& st, const char *filename)
{
  constexpr size_t size = 10000;
  uint32_t *array = new uint32_t[size];

  // Random initialisation
  for (size_t i = 0; i < size; ++i)
    array[i] = rand();

  uint32_t *GPU_keys;
  cudaMalloc(&GPU_keys, sizeof(uint32_t) * size);

  // Is absolutely not used, but is needed for the function
  size_t *GPU_values;
  cudaMalloc(&GPU_values, sizeof(size_t) * size);


  for (auto _ : st)
  {// Copy the array each times as the second times, the GPU_keys is already sorted
    cudaMemcpy(GPU_keys, array, sizeof(uint32_t) * size, cudaMemcpyDefault);
    single_thread_bubble_sort(GPU_keys, GPU_values, size);
  }

  delete[] array;
  cudaFree(GPU_keys);
}

FULL_BENCHMARK(BM_sort_object_bubble_sort);


/*
 * Benchmark the octree node difference computation.
 */
void BM_nodes_difference(benchmark::State& st, const char *filename)
{
  struct scene scene = parser(filename);
  struct scene *cuda_scene = to_cuda(&scene);

  struct scene CPU_scene;
  cudaMemcpy(&CPU_scene, cuda_scene, sizeof(struct scene), cudaMemcpyDefault);

  dim3 threadsPerBlock(32);
  dim3 numBlocks(ceil(scene.object_count * 1.0 / threadsPerBlock.x));

  // Compute the bounding box
  struct AABB *aabbs;
  cudaMalloc(&aabbs, sizeof(struct AABB) * scene.object_count);
  object_compute_bounding_box<<<numBlocks, threadsPerBlock>>>(cuda_scene, aabbs);

  // Compute the global scale
  struct AABB *resulting_scale;
  cudaMalloc(&resulting_scale, sizeof(struct AABB));
  find_scene_scale_shared<<<numBlocks, threadsPerBlock>>>(aabbs, scene.object_count, resulting_scale);

  // Get the position of all objects
  octree_generation_position *positions;
  cudaMalloc(&positions, sizeof(octree_generation_position) * scene.object_count);
  position_object<<<numBlocks, threadsPerBlock>>>(aabbs, resulting_scale, positions, scene.object_count);

  // Sort all objects for easier nodes difference
  single_thread_bubble_sort(positions, CPU_scene.objects, scene.object_count);

  // Compute the nodes difference
  size_t *nodes_difference;
  cudaMalloc(&nodes_difference, sizeof(size_t) * scene.object_count);

  for (auto _ : st)
    nodes_difference_array<<<numBlocks, threadsPerBlock>>>(positions, nodes_difference, scene.object_count);


  cudaFree(aabbs);
  cudaFree(positions);
  cudaFree(nodes_difference);
}

FULL_BENCHMARK(BM_nodes_difference);


/*
 * Benchmark the octree node difference prefix array computation.
 */
void BM_prefix_sum_single_thread(benchmark::State& st)
{
  constexpr size_t size = 10000;
  size_t *array = new size_t[size];

  // Random initialisation
  for (size_t i = 0; i < size; ++i)
    array[i] = rand();

  size_t *GPU_array;
  cudaMalloc(&GPU_array, sizeof(size_t) * size);
  cudaMemcpy(GPU_array, array, sizeof(size_t) * size, cudaMemcpyDefault);


  for (auto _ : st)
    single_thread_prefix_sum(GPU_array, size);

  delete[] array;
  cudaFree(GPU_array);
}

BENCHMARK(BM_prefix_sum_single_thread);


/*
 * Benchmark the octree node difference prefix array computation.
 */
void BM_octree_creation(benchmark::State& st, const char *filename)
{
  struct scene scene = parser(filename);
  struct scene *cuda_scene = to_cuda(&scene);

  struct scene CPU_scene;
  cudaMemcpy(&CPU_scene, cuda_scene, sizeof(struct scene), cudaMemcpyDefault);

  dim3 threadsPerBlock(32);
  dim3 numBlocks(ceil(scene.object_count * 1.0 / threadsPerBlock.x));

  // Compute the bounding box
  struct AABB *aabbs;
  cudaMalloc(&aabbs, sizeof(struct AABB) * scene.object_count);
  object_compute_bounding_box<<<numBlocks, threadsPerBlock>>>(cuda_scene, aabbs);

  // Compute the global scale
  struct AABB *resulting_scale;
  cudaMalloc(&resulting_scale, sizeof(struct AABB));
  find_scene_scale_shared<<<numBlocks, threadsPerBlock>>>(aabbs, scene.object_count, resulting_scale);

  // Get the position of all objects
  octree_generation_position *positions;
  cudaMalloc(&positions, sizeof(octree_generation_position) * scene.object_count);
  position_object<<<numBlocks, threadsPerBlock>>>(aabbs, resulting_scale, positions, scene.object_count);

  // Sort all objects for easier nodes difference
  single_thread_bubble_sort(positions, CPU_scene.objects, scene.object_count);

  // Compute the nodes difference
  size_t *nodes_difference;
  cudaMalloc(&nodes_difference, sizeof(size_t) * scene.object_count);
  nodes_difference_array<<<numBlocks, threadsPerBlock>>>(positions, nodes_difference, scene.object_count);

  single_thread_prefix_sum(nodes_difference, scene.object_count);

  // Create the resulting octree
  size_t nb_nodes;
  cudaMemcpy(&nb_nodes, nodes_difference + (scene.object_count - 1), sizeof(size_t), cudaMemcpyDefault);
  struct octree *octree;
  cudaMalloc(&octree, sizeof(struct octree) * nb_nodes);

  for (auto _ : st)
    create_octree<<<numBlocks, threadsPerBlock>>>(positions, nodes_difference, scene.object_count, resulting_scale, octree);

  cudaFree(aabbs);
  cudaFree(positions);
  cudaFree(nodes_difference);
  cudaFree(octree);
}

FULL_BENCHMARK(BM_octree_creation);


BENCHMARK_MAIN();