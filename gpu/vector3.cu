#include <math.h>
#include <cuda_runtime.h>
#include "vector3.h"


__device__ vector3 vector3_sub(vector3 a, vector3 b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ vector3 vector3_add(vector3 a, vector3 b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
  vector3 ret;
  ret.x = a.x + b.x;
  ret.y = a.y + b.y;
  ret.z = a.z + b.z;
  return ret;
}


__device__ vector3 vector3_cross(vector3 a, vector3 b)
{
  return make_float3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}

__device__ vector3 vector3_scale(vector3 a, float r)
{
  return make_float3(
    r * a.x,
    r * a.y,
    r * a.z
  );
}

__device__ vector3 vector3_normalize(vector3 a)
{
  float root = vector3_length(a);
  return make_float3(
    a.x / root,
    a.y / root,
    a.z / root
  );
}
