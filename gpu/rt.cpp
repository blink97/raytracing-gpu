#include <err.h>

#include "raytracer.h"
#include "parser.h"
#include "printer.h"

#include <iostream>

int main(int argc, char *argv[])
{
  if (argc != 3)
    errx(1, "usage: %s file.svati output.ppm", argv[0]);

#  if defined(LAYOUT_FRAGMENTED)
  std::cout << "Using fragmented layout" << std::endl;
#  elif defined(LAYOUT_AOS)
  std::cout << "Using array of structures (AOS) layout" << std::endl;
#  else /* LAYOUT_SOA */
  std::cout << "Using structure of arrays (SOA) layout" << std::endl;
#  endif

  struct scene scene = parser(argv[1]);
  struct color output[(scene.camera.width + 1) * (scene.camera.height + 1)];

  struct scene *cuda_scene = to_cuda(&scene);

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
