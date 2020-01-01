#ifndef SORT_H
# define SORT_H

/*
 * Sort the given values using the given keys for sorting.
 * Is used by the octree creation, so that objects next to
 * each others in the octree are together after the sort,
 * speeding the octree creation as no expensive search needs
 * to be done, the objects only need to look near them in the sorted array.
 *
 * The keys must also be sorted at the end of the function.
 */

template<typename T>
void single_thread_bubble_sort(uint32_t *keys, T *values, size_t size);


template<typename T>
void parallel_radix_sort(uint32_t *keys, T *values, size_t size);

#include "sort.tuh"

#endif /* !SORT_H */