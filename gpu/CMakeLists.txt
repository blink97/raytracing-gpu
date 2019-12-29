cmake_minimum_required (VERSION 3.13.4)
project (GPGPU CUDA CXX)

# Enable only one of them to set the layout used.
#add_compile_definitions(LAYOUT_FRAGMENTED)
#add_compile_definitions(LAYOUT_AOS)
add_compile_definitions(LAYOUT_SOA)

include(FetchContent)

FetchContent_Declare(
  GoogleBenchmark
  URL https://github.com/google/benchmark/archive/v1.4.1.tar.gz
  )

FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.8.0.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.4.2.tar.gz
  )


if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}") #To allows cuda type definition in .cpp file


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wextra -Wall -I./headers")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


# The thing used for rendering
add_library(renderer)
include_directories(headers)
target_compile_features(renderer PUBLIC cxx_std_14)
target_link_libraries(renderer PRIVATE spdlog::spdlog)

target_sources(renderer PRIVATE
  colors.cpp
  hit.cpp
  light.cpp
  partitioning/aabb.cu
  partitioning/octree.cu
  parse_obj.cpp
  parser.cpp
  printer.cpp
  scene.cu
  ray.cpp
  raytracer.cpp
  thread_arg.cpp
  vector3-extern.cpp
  vector3.cpp
)
set_target_properties(renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(rt rt.cpp)
target_compile_features(rt PUBLIC cxx_std_17)
target_link_libraries(rt PRIVATE spdlog::spdlog renderer)

add_executable(bench bench.cu)
target_compile_features(bench PUBLIC cxx_std_14)
target_link_libraries(bench PRIVATE spdlog::spdlog benchmark renderer)

add_executable(test test.cu)
target_compile_features(test PUBLIC cxx_std_14)
target_link_libraries(test PRIVATE spdlog::spdlog renderer)
