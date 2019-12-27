#include "vector3.h"

struct vector3 vector3_sub(struct vector3 a, struct vector3 b)
{
  struct vector3 ret;
  ret.x = a.x - b.x;
  ret.y = a.y - b.y;
  ret.z = a.z - b.z;
  return ret;
}

struct vector3 vector3_add(struct vector3 a, struct vector3 b)
{
  struct vector3 ret;
  ret.x = a.x + b.x;
  ret.y = a.y + b.y;
  ret.z = a.z + b.z;
  return ret;
}

struct vector3 vector3_cross(struct vector3 a, struct vector3 b)
{
  struct vector3 ret;
  ret.x = a.y * b.z - a.z * b.y;
  ret.y = a.z * b.x - a.x * b.z;
  ret.z = a.x * b.y - a.y * b.x;
  return ret;
}

struct vector3 vector3_scale(struct vector3 a, float r)
{
  struct vector3 ret;
  ret.x = r * a.x;
  ret.y = r * a.y;
  ret.z = r * a.z;
  return ret;
}

struct vector3 vector3_normalize(struct vector3 a)
{
  struct vector3 ret;
  float root = vector3_length(a);
  ret.x = a.x / root;
  ret.y = a.y / root;
  ret.z = a.z / root;
  return ret;
}
