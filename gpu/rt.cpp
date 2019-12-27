#include <err.h>

#include "raytracer.h"
#include "parser.h"
#include "printer.h"

int main(int argc, char *argv[])
{
  if (argc != 3)
    errx(1, "usage: %s file.svati output.ppm", argv[0]);

  struct scene scene = parser(argv[1]);
  struct color output[(scene.camera.width + 1) * (scene.camera.height + 1)];

  raytrace(scene, output);

  FILE *out = open_output(argv[2], scene.camera.width, scene.camera.height);
  for (int j = scene.camera.height; j > 0; j--)
  {
    for (int i = scene.camera.width; i > 0; i--)
    {
      print_color(output[j * scene.camera.width + i], out);
    }
  }
}
