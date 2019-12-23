#include "colors.h"

struct color init_color(float r, float g, float b)
{
  struct color ret;
  ret.r = r * 255;
  if (ret.r > 255)
    ret.r = 255;
  if (ret.r < 0)
    ret.r = 0;
  ret.g = g * 255;
  if (ret.g > 255)
    ret.g = 255;
  if (ret.g < 0)
    ret.g = 0;
  ret.b = b * 255;
  if (ret.b > 255)
    ret.b = 255;
  if (ret.b < 0)
    ret.b = 0;
  return ret;
}

struct color color_add(struct color a, struct color b)
{
  a.r += b.r;
  if (a.r > 255)
    a.r = 255;
  a.g += b.g;
  if (a.g > 255)
    a.g = 255;
  a.b += b.b;
  if (a.b > 255)
    a.b = 255;
  return a;
}

struct color color_mul(struct color a, float coef)
{
  return init_color(a.r / 255 * coef, a.g / 255 * coef, a.b / 255 * coef);
}

struct color color_mul2(struct color a, struct color b)
{
  return init_color((a.r / 255) * (b.r / 255),
                    (a.g / 255) * (b.g / 255),
                    (a.b / 255) * (b.b / 255));
  return a;
}
