#include "utils.h"

__global__ void single_thread_prefix_sum_kernel(size_t *array, size_t size)
{
  if (blockIdx.x * blockDim.x + threadIdx.x > 1)
    return; // Nothing to do here, prefix array is single thread.

  size_t previous = 0;
  for (size_t i = 0; i < size; ++i)
  {
    previous = array[i] + previous;
    array[i] = previous;
  }
}

void single_thread_prefix_sum(size_t *array, size_t size)
{
  single_thread_prefix_sum_kernel<<<1, 1>>>(array, size);
}





// What must be the size of the block that are done per block.
// This also correspond to the number of thread that will be launched,
// So this number must not be above 1024.
// This number os currently small for testing purpose.
#define PREFIX_SUM_BLOCK_SIZE 4

__device__ __host__ size_t get_shared_prefix_sum_buffer_size(size_t size)
{
  // The buffer will contains all level of summation.
  size_t buffer_size = 0;

  // The size of the current level
  size_t current_size = size;
  while ((current_size = ceil((float)current_size / (float)(PREFIX_SUM_BLOCK_SIZE * 2.0))) > 1)
  {
    buffer_size += current_size;
  }

  return buffer_size;
}


// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
//
// Compute the prefix sum kernel of a block only,
// and place the final sum inside the buffer,
// at the current block, so that the buffer could
// have it's prefix sum also computed, etc...
// The buffer always have enough memory to save the information
__global__ void shared_prefix_sum_kernel(size_t *array, size_t *buffer, size_t size)
{
  // The sum inside a block, not the global one.
  __shared__ size_t block_sum[PREFIX_SUM_BLOCK_SIZE * 2];

  size_t thread_index = threadIdx.x;
  size_t array_offset = blockIdx.x * (PREFIX_SUM_BLOCK_SIZE * 2);

  // Initialise the block sum by copying
  // all memory in the shared memory
  size_t a_first = thread_index;
  size_t a_second = thread_index + PREFIX_SUM_BLOCK_SIZE;

  block_sum[a_first] = ((array_offset + a_first < size) ? array[array_offset + a_first] : 0);
  block_sum[a_second] = ((array_offset + a_second < size) ? array[array_offset + a_second] : 0);

  // Do the up-sweep phase
  size_t offset = 1;
  for (size_t d = PREFIX_SUM_BLOCK_SIZE; d > 0; d /= 2, offset *= 2)
  {
    // Sync all threads so that the previous iteration is really done.
    __syncthreads();

    if (thread_index < d)
    {
      size_t src = offset * (2 * thread_index + 1) - 1;
      size_t dst = offset * (2 * thread_index + 2) - 1;

      block_sum[dst] += block_sum[src];
    }
  }


  // Full sum is only defined for the 1 threads, not the others.
  // The last threads to have touched the last element in this array
  // is this one, so no synchronisation is needed.
  size_t full_sum;
  if (thread_index == 0)
  {
    full_sum = block_sum[PREFIX_SUM_BLOCK_SIZE * 2 - 1];
    // The original algorithm perform an exclusive prefix sum,
    // but an inclusive one is needed for us.
    block_sum[PREFIX_SUM_BLOCK_SIZE * 2 - 1] = 0;
  }


  // Do the down-sweep phase
  for (size_t d = 1; d < PREFIX_SUM_BLOCK_SIZE * 2; d *= 2)
  {
    offset /= 2;
    // Sync all threads so that the previous iteration is really done.
    __syncthreads();


    if (thread_index < d)
    {
      size_t src = offset * (2 * thread_index + 1) - 1;
      size_t dst = offset * (2 * thread_index + 2) - 1;


      size_t tmp = block_sum[src];
      block_sum[src] = block_sum[dst];
      block_sum[dst] += tmp;
    }
  }

  __syncthreads();

  // Write back the value in the input prefix array
  // The original algorithm use an exclusive range,
  // correction is done here.
  if (array_offset + a_first < size)
    array[array_offset + a_first] = block_sum[a_first + 1];
  if (array_offset + a_second < size && a_second + 1 < PREFIX_SUM_BLOCK_SIZE * 2)
    array[array_offset + a_second] = block_sum[a_second + 1];


  if (thread_index == 0)
  {
    // Set the last elements, in the prefix array and the buffer
    size_t last_offset = array_offset + PREFIX_SUM_BLOCK_SIZE * 2 - 1;
    if (last_offset < size)
      array[last_offset] = full_sum;

    // If need recursion, the the full sum value in the buffer
    if (PREFIX_SUM_BLOCK_SIZE * 2 < size)
      buffer[blockIdx.x] = full_sum;
  }
}

// Fix the current prefix sum with the next one,
// by adding the resulting prefix sum to all element in the array
__global__ void fix_prefix_sum(size_t *array, size_t *buffer, size_t size)
{
  size_t index = (blockIdx.x + 1) * PREFIX_SUM_BLOCK_SIZE * 2 + threadIdx.x * 2;
  if (index < size)
  {
    array[index] += buffer[blockIdx.x];
    if (index + 1 < size)
    {
      array[index + 1] += buffer[blockIdx.x];
    }
  }
}

void shared_prefix_sum_rec(size_t *array, size_t *buffer, size_t size)
{
  size_t nb_blocks = ceil((float)size / (float)(PREFIX_SUM_BLOCK_SIZE * 2.0));

  shared_prefix_sum_kernel<<<nb_blocks, PREFIX_SUM_BLOCK_SIZE>>>(array, buffer, size);

  if (nb_blocks > 1)
  {
    // Only a partial prefix sum is computed,
    // compute the prefix sum of the partial prefix sum,
    // and then fix the previous prefix sum to have the correct results.
    shared_prefix_sum_rec(buffer, buffer + nb_blocks, nb_blocks);
    fix_prefix_sum<<<
      nb_blocks - 1/* No need to fix the first one */,
      PREFIX_SUM_BLOCK_SIZE
    >>>(array, buffer, size);
  }
}

void shared_prefix_sum(size_t *array, size_t size)
{
  size_t *buffer;
  size_t buffer_size = get_shared_prefix_sum_buffer_size(size);
  cudaMalloc(&buffer, sizeof(size_t) * buffer_size);

  // Launch the recursive function to compute the prefix sum
  shared_prefix_sum_rec(array, buffer, size);

  cudaFree(buffer);
}