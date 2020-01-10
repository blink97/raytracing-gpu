#ifndef PARTITIONING_SORT_H
# define PARTITIONING_SORT_H

/*
 * Sort the given values using the given keys for sorting.
 * Is used by the octree creation, so that objects next to
 * each others in the octree are together after the sort,
 * speeding the octree creation as no expensive search needs
 * to be done, the objects only need to look near them in the sorted array.
 *
 * The keys must also be sorted at the end of the function.
 */

template<typename O, typename A>
__device__ __host__ void bubble_sort(uint32_t *keys, O *values, A *second_values, size_t size);

template<typename O, typename A>
void single_thread_bubble_sort(uint32_t *keys, O *values, A *second_values, size_t size);


template<typename O, typename A>
void parallel_radix_sort(uint32_t *keys, O *values, A *second_values, size_t size);

#include "sort.tuh"

#endif /* !PARTITIONING_SORT_H */