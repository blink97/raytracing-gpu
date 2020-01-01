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


#define PREFIX_SUM_COUNT 100000
#define SORT_COUNT 1000000

/*
 * Benchmark the parser to see memory alignement difference
 */
void BM_parser(benchmark::State& st, const char *filename)
{
  for (auto _ : st)
    parser(filename);
}


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


/*
 * Benchmark the single thread bubble sort
 */
void BM_single_thread_bubble_sort(benchmark::State& st)
{
  constexpr size_t size = SORT_COUNT;
  uint32_t *array = new uint32_t[size];
  uint32_t *keys = new uint32_t[size];
  uint32_t *values = new uint32_t[size];

  // Random initialisation
  for (size_t i = 0; i < size; ++i)
  {
    array[i] = rand();
    values[i] = i;
  }


  for (auto _ : st)
  {// Copy the array each times as the second times, the GPU_keys is already sorted
    cudaMemcpy(keys, array, sizeof(uint32_t) * size, cudaMemcpyDefault);
    bubble_sort(keys, values, size);
  }

  // Assert that the values are sorted
  for (size_t i = 0; i + 1 < size; ++i)
    assert(keys[i] <= keys[i + 1]);

  delete[] array;
  delete[] keys;
  delete[] values;
}


/*
 * Benchmark the single thread bubble sort
 */
void BM_single_thread_stable_sort(benchmark::State& st)
{
  constexpr size_t size = SORT_COUNT;
  uint32_t *array = new uint32_t[size];
  uint32_t *keys = new uint32_t[size];
  uint32_t *values = new uint32_t[size];

  // Random initialisation
  for (size_t i = 0; i < size; ++i)
  {
    array[i] = rand();
    values[i] = i;
  }


  for (auto _ : st)
  {// Copy the array each times as the second times, the GPU_keys is already sorted
    cudaMemcpy(keys, array, sizeof(uint32_t) * size, cudaMemcpyDefault);
    std::stable_sort(keys, keys + size);
  }

  // Assert that the values are sorted
  for (size_t i = 0; i + 1 < size; ++i)
    assert(keys[i] <= keys[i + 1]);

  delete[] array;
  delete[] keys;
  delete[] values;
}


/*
 * Benchmark the parallel radix sort.
 */
void BM_parallel_radix_sort(benchmark::State& st)
{
  constexpr size_t size = SORT_COUNT;
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
    parallel_radix_sort(GPU_keys, GPU_values, size);
  }

  // Assert that the values are sorted
  cudaMemcpy(array, GPU_keys, sizeof(uint32_t) * size, cudaMemcpyDefault);
  for (size_t i = 0; i + 1 < size; ++i)
    assert(array[i]<= array[i + 1]);

  delete[] array;
  cudaFree(GPU_keys);
  cudaFree(GPU_values);
}


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


/*
 * Benchmark the octree node difference prefix array computation.
 */
void BM_prefix_sum_single_thread(benchmark::State& st)
{
  constexpr size_t size = PREFIX_SUM_COUNT;
  size_t *array = new size_t[size];

  // Random initialisation
  size_t sum = 0;
  for (size_t i = 0; i < size; ++i)
  {
    size_t current = rand() % 8;
    array[i] = current;
    sum += current;
  }

  size_t *GPU_array;
  cudaMalloc(&GPU_array, sizeof(size_t) * size);
  cudaMemcpy(GPU_array, array, sizeof(size_t) * size, cudaMemcpyDefault);


  for (auto _ : st)
  {
    cudaMemcpy(GPU_array, array, sizeof(size_t) * size, cudaMemcpyDefault);
    single_thread_prefix_sum(GPU_array, size);
  }

  cudaMemcpy(array, GPU_array, sizeof(size_t) * size, cudaMemcpyDefault);
  assert(array[size - 1] == sum);

  delete[] array;
  cudaFree(GPU_array);
}


/*
 * Benchmark the octree node difference prefix array computation.
 */
void BM_prefix_sum_parallel(benchmark::State& st)
{
  constexpr size_t size = PREFIX_SUM_COUNT;
  size_t *array = new size_t[size];

  // Random initialisation
  size_t sum = 0;
  for (size_t i = 0; i < size; ++i)
  {
    size_t current = rand() % 8;
    array[i] = current;
    sum += current;
  }

  size_t *GPU_array;
  cudaMalloc(&GPU_array, sizeof(size_t) * size);
  cudaMemcpy(GPU_array, array, sizeof(size_t) * size, cudaMemcpyDefault);


  for (auto _ : st)
  {
    cudaMemcpy(GPU_array, array, sizeof(size_t) * size, cudaMemcpyDefault);
    shared_prefix_sum(GPU_array, size);
  }


  cudaMemcpy(array, GPU_array, sizeof(size_t) * size, cudaMemcpyDefault);
  assert(array[size - 1] == sum);

  delete[] array;
  cudaFree(GPU_array);
}


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


FULL_BENCHMARK(BM_parser);
FULL_BENCHMARK(BM_aabb_object);
FULL_BENCHMARK(BM_find_scene_scale_basic);
FULL_BENCHMARK(BM_find_scene_scale_shared);
FULL_BENCHMARK(BM_position_object);

FULL_BENCHMARK(BM_nodes_difference);
FULL_BENCHMARK(BM_octree_creation);

// Sorting
BENCHMARK(BM_single_thread_bubble_sort);
BENCHMARK(BM_single_thread_stable_sort);
BENCHMARK(BM_parallel_radix_sort);

// Prefix sum
BENCHMARK(BM_prefix_sum_single_thread);
BENCHMARK(BM_prefix_sum_parallel);

BENCHMARK_MAIN();