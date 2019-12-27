#include "printer.h"

FILE *open_output(const char *output, int width, int height)
{
  FILE *out = fopen(output, "w+");
  if (!out)
    errx(1, "%s\n", strerror(errno));
  fprintf(out, "P3\n%d %d\n255\n", width, height);
  return out;
}

void print_color(struct color color, FILE *output)
{
  int r = color.r;
  int g = color.g;
  int b = color.b;
  fprintf(output, "%d %d %d ", r, g, b);
}
