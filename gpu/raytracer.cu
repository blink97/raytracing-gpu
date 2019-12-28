__global__ void raytrace(char* buff, int width, int height, size_t pitch)
{
  int px = blockDim.x * blockIdx.x + threadIdx.x;
  int py = blockDim.y * blockIdx.y + threadIdx.y;

  if (px >=. width || py >= height)
    return

  uint32_t lineptr = (uint32_t*)(buff + py * pitch);

  lineptr[px] = color
}


void render(const scene &scene, char* buffer, int aliasing)
{
  vector3 u = vector3_normalize(scene.camera.u);
  vector3 v = vector3_normalize(scene.camera.v);
  vector3 w = vector3_cross(u, v);
  float L = scene.camera.width / (2 * tan(scene.camera.fov * M_PI / 360));
  vector3 C = vector3_add(scene.camera.position, vector3_scale(w, L));

  int width = scene.camera.width;
  int height = scene.camera.height;

  float aliasing_step = 1.0 / aliasing;

  for (int x = 0; x < width; x++)
  {
    for (int y = 0; y < height; y++)
    {
      struct color color = init_color(0, 0, 0);
      vector3 ui = vector3_scale(u, x);
      vector3 vj = vector3_scale(v, y);
      vector3 point = vector3_add(vector3_add(C, ui), vj);
      vector3 direction = vector3_normalize(vector3_sub(scene.camera.position, point));
      struct ray ray;
      ray.origin = point;
      ray.direction = direction;
      struct color tcolor = trace(scene, ray, 1);
      tcolor = color_mul(tcolor, 0.25);
      color = color_add(color, tcolor);
      output[y * scene.camera.width + x] = color;
    }
  }

  cudaError_t rc = cudaSuccess;
  // Allocate device memory
  char*  devBuffer;
  size_t pitch;
  uchar4* LUT;

  rc = cudaMalloc(&LUT, (n_iterations + 1) * sizeof(uchar4));
  if (rc)
    abortError("Fail LUT allocation");
  rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(rgba8_t), height);
  if (rc)
    abortError("Fail buffer allocation");

  // Run the kernel with blocks of size 64 x 64
  {
    int bsize = 32;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);
    apply_LUT<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch, n_iterations, LUT);

    if (cudaPeekAtLastError())
      abortError("Computation Error");
  }

  // Copy back to main memory
  rc = cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost);
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(devBuffer);
  if (rc)
    abortError("Unable to free memory");
}
