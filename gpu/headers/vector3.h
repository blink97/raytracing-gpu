#ifndef VECTOR3_H
# define VECTOR3_H

#include <cuda_runtime.h>
//#include "vector_types.h"

typedef float3 vector3;

/* custom ope */
__device__ float3 operator+(const float3 &a, const float3 &b);
__device__ float3 operator-(const float3 &a, const float3 &b);
__device__ float operator~(const float3 &a); // LENGTH !


/* vector3.c */

__device__ vector3 vector3_sub(vector3 a, vector3 b);
__device__ vector3 vector3_add(vector3 a, vector3 b);
__device__ vector3 vector3_cross(vector3 a, vector3 b);
__device__ vector3 vector3_scale(vector3 a, float r);
__device__ vector3 vector3_normalize(vector3 a);

/* vector3-extern.c */

__device__ float vector3_dot(vector3 a, vector3 b);
__device__ float vector3_length(vector3 a);
__device__ int vector3_cmp(vector3 a, vector3 b);
__device__ int vector3_is_zero(vector3 a);

#endif /* !VECTOR3_H */
