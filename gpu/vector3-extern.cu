#include <math.h>

#include "vector3.h"

__host__ __device__ float vector3_dot(vector3 a, vector3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float vector3_length(vector3 a)
{
  return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__ int vector3_cmp(vector3 a, vector3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

__host__ __device__ int vector3_is_zero(vector3 a)
{
  return a.x == 0 && a.y == 0 && a.z == 0;
}
