#include "utils.h"

__device__ float atomicMinFloat(float *addr, float value)
{
  float old = ((value >= 0)
    ? __int_as_float(atomicMin((int *)addr, __float_as_int(value)))
    : __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value))));

  return old;
}

__device__ float atomicMaxFloat(float *addr, float value) {
  float old = ((value >= 0)
    ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
    : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value))));

    return old;
}

__host__ __device__ size_t binary_search(const size_t *const array, size_t size, size_t value)
{
  size_t left = 0;
  size_t right = size;

  while (left < right)
  {
    // equivalent to (left + right) / 2,
    // is changed to prevent overflowing.
    size_t m = left + (right - left) / 2;

    if (array[m] < value + 1)
      left = m + 1;
    else
      right = m;
  }

  return left;
}