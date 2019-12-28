#include <benchmark/benchmark.h>

#include "parser.h"
#include "partitioning/aabb.h"

#define TESTS_PATH "../../tests/"

// All tests
#define CUBE TESTS_PATH "cube.svati"
#define ISLAND_SMOOTH TESTS_PATH "island_smooth.svati"
#define SPHERES TESTS_PATH "spheres.svati"

/*
 * Benchmark the parser to see memory alignement difference
 */
template <class ...ExtraArgs>
void BM_parser(benchmark::State& st, ExtraArgs&&... extra_args)
{
  for (auto _ : st)
    parser(std::forward<ExtraArgs>(extra_args)...);
}

//BENCHMARK_CAPTURE(BM_parser, simple_cube, CUBE);
//BENCHMARK_CAPTURE(BM_parser, medium_island, ISLAND_SMOOTH);
//BENCHMARK_CAPTURE(BM_parser, complex_sphere, SPHERES);

/*
 * Benchmark the AABB creation with the object strategy
 */
void BM_aabb_object(benchmark::State& st)
{
  struct scene scene = parser(SPHERES);
  struct scene cuda_scene = to_cuda(&scene);

  struct AABB *aabbs;
  cudaMalloc(&aabbs, sizeof(struct AABB) * cuda_scene.object_count);

  for (auto _ : st)
    compute_bounding_box(&cuda_scene, aabbs);
}

BENCHMARK(BM_aabb_object);


BENCHMARK_MAIN();