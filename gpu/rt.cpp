#include <err.h>

#include "raytracer.h"

int main(int argc, char *argv[])
{
  if (argc != 3)
    errx(1, "usage: %s file.svati output.ppm", argv[0]);
  raytrace(argv[1], argv[2]);
}
