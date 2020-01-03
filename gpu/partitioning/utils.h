#ifndef PARTITIONING_UTILS_H
# define PARTITIONING_UTILS_H

__device__ float atomicMinFloat(float *addr, float value);

__device__ float atomicMaxFloat(float *addr, float value);

__host__ __device__ size_t binary_search(const size_t *const array, size_t size, size_t value);

#endif /* !PARTITIONING_UTILS_H */