#include <benchmark/benchmark.h>

#include "parser.h"

#define TESTS_PATH "../../tests/"

// All tests
#define SIMPLE_CUBE TESTS_PATH "cube.svati"
#define MEDIUM_ISLAND TESTS_PATH "island_smooth.svati"
#define COMPLEX_SPHERE TESTS_PATH "spheres.svati"

template <class ...ExtraArgs>
void BM_parser(benchmark::State& st, ExtraArgs&&... extra_args)
{
  for (auto _ : st)
    parser(std::forward<ExtraArgs>(extra_args)...);
}

BENCHMARK_CAPTURE(BM_parser, simple_cube, SIMPLE_CUBE);
BENCHMARK_CAPTURE(BM_parser, medium_island, MEDIUM_ISLAND);
BENCHMARK_CAPTURE(BM_parser, complex_sphere, COMPLEX_SPHERE);


BENCHMARK_MAIN();