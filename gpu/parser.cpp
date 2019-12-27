#include "parser.h"
#include "parse_obj.h"

#include <cstdlib>

struct camera parse_camera(FILE *file)
{
  struct camera cam;
  fscanf(file, "%d %d %f %f %f %f %f %f %f %f %f %f",
        &cam.width,
        &cam.height,
        &cam.position.x,
        &cam.position.y,
        &cam.position.z,
        &cam.u.x,
        &cam.u.y,
        &cam.u.z,
        &cam.v.x,
        &cam.v.y,
        &cam.v.z,
        &cam.fov);
  return cam;
}

struct light parse_a_light(FILE *file)
{
  struct light light;
  light.type = AMBIENT;
  fscanf(file, "%f %f %f",
        &light.r,
        &light.g,
        &light.b);
  return light;
}

struct light parse_d_light(FILE *file)
{
  struct light light;
  light.type = DIRECTIONAL;
  fscanf(file, "%f %f %f %f %f %f",
        &light.r,
        &light.g,
        &light.b,
        &light.v.x,
        &light.v.y,
        &light.v.z);
  return light;
}

struct light parse_p_light(FILE *file)
{
  struct light light;
  light.type = POINT;
  fscanf(file, "%f %f %f %f %f %f",
        &light.r,
        &light.g,
        &light.b,
        &light.v.x,
        &light.v.y,
        &light.v.z);
  return light;
}

struct scene parser(const char *path)
{
  struct scene scene;
  scene.object_count = 0;
  scene.light_count = 0;
  struct object *objects = (struct object *)malloc(sizeof (struct object));
  struct light *lights = (struct light *)malloc(sizeof (struct light));
  FILE *svati = fopen(path, "r");
  if (!svati)
    errx(1, "%s\n", strerror(errno));
  char instruction[81];
  while (fscanf(svati, "%s", instruction) != EOF)
  {
    if (strcmp(instruction, "camera") == 0)
    {
      struct camera cam = parse_camera(svati);
      scene.camera = cam;
    }
    else if (strcmp(instruction, "a_light") == 0)
    {
      struct light a_light = parse_a_light(svati);
      scene.light_count++;
      lights = (struct light *)realloc(lights, sizeof (struct light) * scene.light_count);
      lights[scene.light_count - 1] = a_light;
    }
    else if (strcmp(instruction, "p_light") == 0)
    {
      struct light p_light = parse_p_light(svati);
      scene.light_count++;
      lights = (struct light *)realloc(lights, sizeof (struct light) * scene.light_count);
      lights[scene.light_count - 1] = p_light;
    }
    else if (strcmp(instruction, "d_light") == 0)
    {
      struct light d_light = parse_d_light(svati);
      scene.light_count++;
      lights = (struct light *)realloc(lights, sizeof (struct light) * scene.light_count);
      lights[scene.light_count - 1] = d_light;
    }
    else if (strcmp(instruction, "object") == 0)
    {
      struct object object = parse_object(svati);
      scene.object_count++;
      objects = (struct object *)realloc(objects, sizeof (struct object) * scene.object_count);
      objects[scene.object_count - 1] = object;
    }
    else if (strcmp(instruction, "#") == 0)
      fscanf(svati, " %[^\n]", instruction);
    else
      errx(1, "Error during the parsing %s\n", instruction);
  }
  scene.objects = objects;
  scene.lights = lights;
  return scene;
}
