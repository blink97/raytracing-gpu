#ifndef VECTOR3_H
# define VECTOR3_H

struct vector3 {
  float x;
  float y;
  float z;
};

/* vector3.c */

struct vector3 vector3_sub(struct vector3 a, struct vector3 b);
struct vector3 vector3_add(struct vector3 a, struct vector3 b);
struct vector3 vector3_cross(struct vector3 a, struct vector3 b);
struct vector3 vector3_scale(struct vector3 a, float r);
struct vector3 vector3_normalize(struct vector3 a);

/* vector3-extern.c */

float vector3_dot(struct vector3 a, struct vector3 b);
float vector3_length(struct vector3 a);
int vector3_cmp(struct vector3 a, struct vector3 b);
int vector3_is_zero(struct vector3 a);

#endif /* !VECTOR3_H */
