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

CUDA_HOSTDEV struct color init_color(float r, float g, float b);
CUDA_HOSTDEV struct color color_add(struct color a, struct color b);
CUDA_HOSTDEV struct color color_mul(struct color a, float coef);
CUDA_HOSTDEV struct color color_mul2(struct color a, struct color b);

#endif /* !COLORS_H */
