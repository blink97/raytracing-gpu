#include "colors.h"

__device__ struct color init_color(float r, float g, float b)
{
  r = r * 255;
  if (r > 255)
    r = 255;
  if (r < 0)
    r = 0;
  g = g * 255;
  if (g > 255)
    g = 255;
  if (g < 0)
    g = 0;
  b = b * 255;
  if (b > 255)
    b = 255;
  if (b < 0)
    b = 0;
  return color{(unsigned char)r, (unsigned char)g, (unsigned char)b, 255};
}

__device__ struct color color_add(struct color* a, struct color* b)
{
  auto r = a->r + b->r;
  if (r > 255)
    r = 255;
  auto g = a->g + b->g;
  if (g > 255)
    g = 255;
  auto blu = a->b + b->b;
  if (blu > 255)
    blu = 255;
  return color{(unsigned char)r, (unsigned char)g, (unsigned char)blu, 255};
}

__device__ struct color color_mul(struct color* a, float coef)
{
  return init_color(float(a->r) / 255 * coef, float(a->g) / 255 * coef, float(a->b) / 255 * coef);
}

__device__ struct color color_mults(struct color a, struct color b)
{
  return init_color((float(a.r) / 255) * (float(b.r) / 255),
                    (float(a.g) / 255) * (float(b.g) / 255),
                    (float(a.b) / 255) * (float(b.b) / 255));
}


