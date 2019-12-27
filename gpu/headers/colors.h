#ifndef COLORS_H
# define COLORS_H

struct color {
  float r;
  float g;
  float b;
};

struct color init_color(float r, float g, float b);
struct color color_add(struct color a, struct color b);
struct color color_mul(struct color a, float coef);
struct color color_mul2(struct color a, struct color b);

#endif /* !COLORS_H */
