#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <spdlog/spdlog.h>
#include <cassert>

#include "raytracer.h"
#include "vector3.h"
#include "ray.h"
#include "colors.h"
#include "hit.h"
#include "light.h"
#include "thread_arg.h"
//#include "ray_color.h"


[[gnu::noinline]]
void _abortError(const char *msg, const char *fname, int line) {
    cudaError_t err = cudaGetLastError();
    spdlog::error("{} ({}, line: {})", msg, fname, line);
    spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
    std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)


__device__ static struct color
trace(struct scene *scene, struct object *objects, struct ray ray, struct ray *reflection, float *loc_nr) {
    struct object obj;
    struct ray new_ray = collide(scene, objects, ray, &obj);
    if (!vector3_is_zero(new_ray.direction)) {
        struct color object = apply_light(scene, objects, &obj, new_ray);
        if (obj.nr > 0) {
            struct ray reflection_ray = ray_bounce(&ray, &new_ray);
            *reflection = reflection_ray;
        }
        *loc_nr = obj.nr;
        return object;
    }
    *loc_nr = 0; // avoid infinte loop ?
    return init_color(0, 0, 0);
}

__global__ void downscale(char* higher, char* lower, int width, int height, size_t pitch, size_t b_pitch, int oh, int ow, int aliasing) {
    // lower buffer position
    int px = blockDim.x * blockIdx.x + threadIdx.x;
    int py = blockDim.y * blockIdx.y + threadIdx.y;

    if (px >= width || py >= height)
        return;

    // higher buffer position
    int m_px = aliasing * px + aliasing;
    int m_py = aliasing * py + aliasing;

    float r = 0;
    float g = 0;
    float b = 0;


    for (int h_py = aliasing * py; h_py < m_py; ++h_py) {
        struct color *local_line = (struct color *) (higher + (oh - h_py - 1) * b_pitch);
        for (int h_px = aliasing * px; h_px < m_px; ++h_px) {
            struct color tmp = local_line[ow - h_px - 1];
            r += (float)tmp.r;
            g += (float)tmp.g;
            b += (float)tmp.b;
        }
    }

    float ali2 = 255.0f * (float)aliasing * (float)aliasing;
    r /= ali2;
    g /= ali2;
    b /= ali2;

    struct color end_color = init_color(r, g, b);

    struct color *lineptr = (struct color *) (lower + (height - py - 1) * pitch);
    lineptr[width - px - 1] = end_color;
}

__global__ void raytrace(char *buff, int width, int height, size_t pitch,
                         struct scene *scene, vector3 *u, vector3 *v, vector3 *C) {

    __shared__ struct object objects[12];
    __shared__ struct scene s_scene[1];

    // Buffer position
    int px = blockDim.x * blockIdx.x + threadIdx.x;
    int py = blockDim.y * blockIdx.y + threadIdx.y;

    if (px >= width || py >= height)
        return;

    for (int t = 0; t < scene->object_count; t++)
      objects[t] = scene->objects[t];

    s_scene[0] = scene[0];

    __syncthreads();

    uint32_t *lineptr = (uint32_t *) (buff + (height - py - 1) * pitch);

    vector3 ui = vector3_scale(*u, px - width / 2);
    vector3 vj = vector3_scale(*v, py - height / 2);

    vector3 point = vector3_add(vector3_add(*C, ui), vj);
    vector3 direction = vector3_normalize(vector3_sub(scene->camera.position, point));
    struct ray ray;


    float nr_ = 1;
    float r_rn = nr_;
    ray.origin = point;
    ray.direction = direction;


    int MAX_BOUNCE = 10; // 100% reflective reflecting each others !..
    struct color color = init_color(0,0,0);
    struct color tmp_color;
    do {
        tmp_color = trace(&s_scene[0], objects, ray, &ray, &r_rn);
        tmp_color = color_mul(&tmp_color, nr_);
        color = color_add(&color, &tmp_color);
        nr_ *= r_rn;

    } while (nr_ > 0.01f && MAX_BOUNCE-- > 0);


//  struct color color = trace(scene, ray, 1);
//  struct color color = init_color(0, 1, 0);

    lineptr[width - px - 1] = *(uint32_t *) &color;
}

float host_v_len(vector3 a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

vector3 host_v_norm(vector3 a) {
    float root = host_v_len(a);
    return make_float3(
            a.x / root,
            a.y / root,
            a.z / root
    );
}

vector3 host_v_cross(vector3 a, vector3 b) {
    return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
    );
}

vector3 host_v_add(vector3 a, vector3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    vector3 ret;
    ret.x = a.x + b.x;
    ret.y = a.y + b.y;
    ret.z = a.z + b.z;
    return ret;
}

vector3 host_v_scale(vector3 a, float r) {
    return make_float3(
            r * a.x,
            r * a.y,
            r * a.z
    );
}

// HOST CODE
void render_loop() {
    // While relfections :
    // Get color and shade with bounce direction and reflection factor
    // If bounced, get color from reflections; loop

}

void render(const scene &scene, char *buffer, int aliasing, std::ptrdiff_t stride, int pre_h, int pre_w) {
    vector3 u = host_v_norm(scene.camera.u);
    vector3 v = host_v_norm(scene.camera.v);
    vector3 w = host_v_cross(u, v);
    float L = scene.camera.width / (2 * tan(scene.camera.fov * M_PI / 360));
    vector3 C = host_v_add(scene.camera.position, host_v_scale(w, L));

    int width = scene.camera.width;
    int height = scene.camera.height;

    cudaError_t rc = cudaSuccess;
    // Allocate device memory
    char *devBuffer;
    size_t pitch;

    // rc = cudaMalloc(&LUT, (n_iterations + 1) * sizeof(uchar4));
    rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(struct color), height);
    if (rc)
        abortError("Fail buffer allocation");

    // Run the kernel with blocks of size 64 x 64
    int bsize = 16;
    int wi = std::ceil((float) width / bsize);
    int he = std::ceil((float) height / bsize);


    spdlog::debug("running kernel of size ({},{})", wi, he);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(wi, he);

    vector3 *cuda_u;
    cudaMalloc(&cuda_u, sizeof(vector3));
    vector3 *cuda_v;
    cudaMalloc(&cuda_v, sizeof(vector3));
    vector3 *cuda_C;
    cudaMalloc(&cuda_C, sizeof(vector3));

    cudaMemcpy(cuda_u, &u, sizeof(vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_v, &v, sizeof(vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_C, &C, sizeof(vector3), cudaMemcpyHostToDevice);


    struct scene *cuda_scene = to_cuda(&scene);
    printf("lancement. %i %i %i %i.\n", wi, he, width, height);
    printf("nb obj, %d\n", scene.object_count);

    raytrace <<< dimGrid, dimBlock>>> (devBuffer, width, height, pitch, cuda_scene, cuda_u, cuda_v, cuda_C);
    // devBuffer now contains the upscaled image

    char *lowerscale;
    size_t second_picth;
    cudaMallocPitch(&lowerscale, &second_picth, pre_w * sizeof(struct color) , pre_h);
    // Dont care for now, for out of range
    downscale<<<dimGrid, dimBlock>>>(devBuffer, lowerscale, pre_w, pre_h, second_picth, pitch, height, width, aliasing);

//    cudaDeviceSynchronize();
    printf("done..\n");

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    // Free
    rc = cudaFree(devBuffer);
    if (rc)
        abortError("Unable to free memory");

        // Copy back to main memory
    rc = cudaMemcpy2D(buffer, stride, lowerscale, second_picth, pre_w * sizeof(struct color), pre_h, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(lowerscale);
    if (rc)
        abortError("Unable to free memory");
}
