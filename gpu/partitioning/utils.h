#ifndef PARTITIONING_UTILS_H
# define PARTITIONING_UTILS_H

/*
 * Compute the prefix sum array of an array.
 *
 * This allows to simply knows how many octree nodes must be
 * created in advance, and allowing fast octree construction,
 * as the octree can be fully constructed in parallel.
 * The number of octree nodes that need to be created
 * is the last value in the array.
 */
void single_thread_prefix_sum(size_t *array, size_t size);

void shared_prefix_sum(size_t *array, size_t size);

#endif /* !PARTITIONING_UTILS_H */