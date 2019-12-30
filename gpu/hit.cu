#include <cuda_runtime.h>
#include "hit.h"
#include "vector3.h"


__device__ static int ray_intersect(struct ray ray, vector3 *input_vertex, vector3 *input_normal,
                         vector3 *out, vector3 *normal)
{
  const float EPSILON = 0.0000001;
  vector3 vertex0 = input_vertex[0];
  vector3 vertex1 = input_vertex[1];
  vector3 vertex2 = input_vertex[2];
  vector3 normal0 = vector3_normalize(input_normal[0]);
  vector3 normal1 = vector3_normalize(input_normal[1]);
  vector3 normal2 = vector3_normalize(input_normal[2]);
  vector3 edge1, edge2, h, s, q;
  float a, f, u, v;
  edge1 = vector3_sub(vertex1, vertex0);
  edge2 = vector3_sub(vertex2, vertex0);
  h = vector3_cross(ray.direction, edge2);
  a = vector3_dot(edge1, h);
  if (a > -EPSILON && a < EPSILON)
    return 0;
  f = 1 / a;
  s = vector3_sub(ray.origin, vertex0);
  u = f * vector3_dot(s, h);
  if (u < 0.0 || u > 1.0)
    return 0;
  q = vector3_cross(s, edge1);
  v = f * vector3_dot(ray.direction, q);
  if (v < 0.0 || u + v > 1.0)
    return 0;

  float t = f * vector3_dot(edge2, q);
  if (t > EPSILON)
  {
    vector3 t2 = vector3_scale(vector3_normalize(ray.direction),
                                      t * vector3_length(ray.direction));
    *out = vector3_add(ray.origin, t2);
    *normal = vector3_add(vector3_add(vector3_scale(normal0, 1 - u - v),
                                    vector3_scale(normal1, u)),
                         vector3_scale(normal2, v));
    return 1;
  }
  return 0;
}

__device__ static struct ray triangle_collide(struct object object, struct ray ray)
{
  float distance = 0;
  struct ray ret = init_ray();
  for (size_t i = 0; i < object.triangle_count; i++)
  {
    vector3 out;
    vector3 normal;
    int has_intersected = ray_intersect(
      ray,
      get_vertex(object.triangles, i), get_normal(object.triangles, i),
      &out, &normal
    );

    if (has_intersected)
    {
      float new_dist = vector3_length(vector3_sub(out, ray.origin));
      if (new_dist > 0.01 && (new_dist < distance || distance == 0))
      {
        distance = new_dist;
        struct ray new_ret;
        new_ret.origin = out;
        new_ret.direction = normal;
        ret = new_ret;
      }
    }
  }
  return ret;
}

__device__ struct ray collide(struct scene* scene, struct ray ray, struct object* obj)
{
  float distance = 0;
  struct ray ret = init_ray();
  for (size_t i = 0; i < scene->object_count; i++)
  {
    struct ray new_ray = triangle_collide(scene->objects[i], ray);
    if (!vector3_is_zero(new_ray.direction))
    {
      float new_dist = vector3_length(vector3_sub(new_ray.origin, ray.origin));
      if (new_dist > 0.01 && (new_dist < distance || distance == 0))
      {
        distance = new_dist;
        ret = new_ray;
        *obj = scene->objects[i];
      }
    }
  }
  return ret;
}

__device__ float3 operator+(const float3 &a, const float3 &b) {

  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}

__device__ float3 operator-(const float3 &a, const float3 &b) {

  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);

}


__device__ float operator~(const float3 &a) {
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}


__device__ float collide_dist(struct scene* scene, struct ray ray)
{
  float distance = 0;
  for (size_t i = 0; i < scene->object_count; i++)
  {
    struct ray new_ray = triangle_collide(scene->objects[i], ray);
    if (!vector3_is_zero(new_ray.direction))
    {
//    	make_float3(new_ray.origin.x - ray.origin.x, new_ray.origin.y - ray.origin.y, new_ray.origin.z - ray.origin.z);
//    	auto res = new_ray.origin - ray.origin;
    	vector3 res = vector3_sub(ray.origin, ray.origin);
    	float new_dist = vector3_length(res);
//    	float new_dist = ~res;

      if (new_dist > 0.01 && (new_dist < distance || distance == 0))
      {
        distance = new_dist;
      }
    }
  }
  return distance;
}
