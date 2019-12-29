#include <benchmark/benchmark.h>
#include <functional>

#include "parser.h"
#include "partitioning/aabb.h"
#include "partitioning/octree.h"

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

FULL_BENCHMARK(BM_aabb_object);


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


BENCHMARK_MAIN();