#include <cuda_runtime.h>

#ifndef COLORS_H
# define COLORS_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <stdint.h>

struct color {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
};

//__host__ __device__ struct color operator*(struct color &a, struct color &b);
//__host__ __device__ struct color operator*(struct color &a, float b);

__device__ struct color init_color(float r, float g, float b);
__device__ struct color color_add(struct color* a, struct color* b);
__device__ struct color color_mul(struct color* a, float coef);
__device__ struct color color_mul2(struct color* a, struct color* b);

#endif /* !COLORS_H */
