#include <err.h>
#include <png.h>

#include "raytracer.h"
#include "parser.h"
#include "printer.h"
#include "colors.h"

#include <iostream>
#include <memory>

void write_png(const std::byte* buffer,
               int width,
               int height,
               int stride,
               const char* filename)
{
  png_structp png_ptr =
    png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr)
    return;

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
    png_destroy_write_struct(&png_ptr, nullptr);
    return;
  }

  FILE* fp = fopen(filename, "wb");
  png_init_io(png_ptr, fp);

  png_set_IHDR(png_ptr, info_ptr,
               width,
               height,
               8,
               PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);
  for (int i = 0; i < height; ++i)
  {
    png_write_row(png_ptr, reinterpret_cast<png_const_bytep>(buffer));
    buffer += stride;
  }

  png_write_end(png_ptr, info_ptr);
  png_destroy_write_struct(&png_ptr, nullptr);
  fclose(fp);
}

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

  // struct color output[(scene.camera.width + 1) * (scene.camera.height + 1)];
  
  // Create buffer
  int stride = scene.camera.width * sizeof(struct color);
  auto buffer = std::make_unique<std::byte[]>(scene.camera.height * stride);

  render(scene, reinterpret_cast<char*>(buffer.get()), 1, stride);

  /*FILE *out = open_output(argv[2], scene.camera.width, scene.camera.height);
  for (int j = scene.camera.height; j > 0; j--)
  {
    for (int i = scene.camera.width; i > 0; i--)
    {
      print_color(output[j * scene.camera.width + i], out);
    }
  }*/

  // Save
  write_png(buffer.get(), scene.camera.width, scene.camera.height, stride, argv[2]);
}
