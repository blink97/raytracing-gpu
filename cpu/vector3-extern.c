#include <math.h>

#include "vector3.h"

float vector3_dot(struct vector3 a, struct vector3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

float vector3_length(struct vector3 a)
{
  return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

int vector3_cmp(struct vector3 a, struct vector3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

int vector3_is_zero(struct vector3 a)
{
  return a.x == 0 && a.y == 0 && a.z == 0;
}
