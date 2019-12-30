#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <spdlog/spdlog.h>
#include <cassert>

#include "raytracer.h"
#include "vector3.h"
#include "ray.h"
#include "colors.h"
#include "hit.h"
#include "light.h"
#include "thread_arg.h"

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)


__device__ static struct color trace(struct scene scene, struct ray ray, float coef)
{
  if (coef < 0.01)
    return init_color(0, 0, 0);
  struct object obj;
  struct ray new_ray = collide(scene, ray, &obj);
  if (!vector3_is_zero(new_ray.direction))
  {
    struct color object = apply_light(scene, obj, new_ray);
    struct ray reflection_ray = ray_bounce(ray, new_ray);
    struct color reflection = trace(scene, reflection_ray, obj.nr * coef);
    struct color ret = color_add(reflection, color_mul(object, coef));
    return ret;
  }
  return init_color(0, 0, 0);
}


__global__ void raytrace(char* buff, int width, int height, size_t pitch,
                         struct scene* scene, vector3* u, vector3* v, vector3* C)
{

  int px = blockDim.x * blockIdx.x + threadIdx.x;
  int py = blockDim.y * blockIdx.y + threadIdx.y;

  if (px >= width || py >= height)
    return;

  uint32_t* lineptr = (uint32_t*)(buff + py * pitch);

  vector3 ui = vector3_scale(*u, px);
  vector3 vj = vector3_scale(*v, py);
  vector3 point = vector3_add(vector3_add(*C, ui), vj);
  vector3 direction = vector3_normalize(vector3_sub(scene->camera.position, point));
  struct ray ray;
  ray.origin = point;
  ray.direction = direction;
  struct color color = trace(*scene, ray, 1);

  lineptr[px] = *(uint32_t *)&color;
}


void render(const scene &scene, char* buffer, int aliasing, std::ptrdiff_t stride)
{
  vector3 u = vector3_normalize(scene.camera.u);
  vector3 v = vector3_normalize(scene.camera.v);
  vector3 w = vector3_cross(u, v);
  float L = scene.camera.width / (2 * tan(scene.camera.fov * M_PI / 360));
  vector3 C = vector3_add(scene.camera.position, vector3_scale(w, L));

  int width = scene.camera.width;
  int height = scene.camera.height;

  float aliasing_step = 1.0 / aliasing;

  cudaError_t rc = cudaSuccess;
  // Allocate device memory
  char*  devBuffer;
  size_t pitch;

  // rc = cudaMalloc(&LUT, (n_iterations + 1) * sizeof(uchar4));
  rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(uchar4), height);
  if (rc)
    abortError("Fail buffer allocation");

  // Run the kernel with blocks of size 64 x 64
  int bsize = 16;
  int wi     = std::ceil((float)width / bsize);
  int he     = std::ceil((float)height / bsize);


  spdlog::debug("running kernel of size ({},{})", wi, he);

  dim3 dimBlock(bsize, bsize);
  dim3 dimGrid(wi, he);

  vector3* cuda_u;
  cudaMalloc(&cuda_u, sizeof(vector3));
  vector3* cuda_v;
  cudaMalloc(&cuda_v, sizeof(vector3));
  vector3* cuda_C;
  cudaMalloc(&cuda_C, sizeof(vector3));

  cudaMemcpy(cuda_u, &u, sizeof(vector3), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_v, &v, sizeof(vector3), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_C, &C, sizeof(vector3), cudaMemcpyHostToDevice);


  struct scene* cuda_scene = to_cuda(&scene);
  printf("lancement. %i %i %i %i.\n", wi, he, width, height);
  raytrace<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch, cuda_scene, cuda_u, cuda_v, cuda_C);
  printf("done..\n");

  //cudaDeviceSynchronize();

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  // Copy back to main memory
  rc = cudaMemcpy2D(buffer, stride, devBuffer, pitch, width * sizeof(struct color), height, cudaMemcpyDeviceToHost);
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(devBuffer);
  if (rc)
    abortError("Unable to free memory");
}
