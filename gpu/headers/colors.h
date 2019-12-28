#ifndef COLORS_H
# define COLORS_H

__device__ struct color {
  float r;
  float g;
  float b;
};

__device__ struct color init_color(float r, float g, float b);
__device__ struct color color_add(struct color a, struct color b);
__device__ struct color color_mul(struct color a, float coef);
__device__ struct color color_mul2(struct color a, struct color b);

#endif /* !COLORS_H */
