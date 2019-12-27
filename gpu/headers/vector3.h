#ifndef VECTOR3_H
# define VECTOR3_H

#include <cuda_runtime.h>
//#include "vector_types.h"

typedef float3 vector3;

/* vector3.c */

vector3 vector3_sub(vector3 a, vector3 b);
vector3 vector3_add(vector3 a, vector3 b);
vector3 vector3_cross(vector3 a, vector3 b);
vector3 vector3_scale(vector3 a, float r);
vector3 vector3_normalize(vector3 a);

/* vector3-extern.c */

float vector3_dot(vector3 a, vector3 b);
float vector3_length(vector3 a);
int vector3_cmp(vector3 a, vector3 b);
int vector3_is_zero(vector3 a);

#endif /* !VECTOR3_H */
