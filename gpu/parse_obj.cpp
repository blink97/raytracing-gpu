#include "parse_obj.h"

struct object init_object(void)
{
  struct object object;
  object.ka.x = 0;
  object.ka.y = 0;
  object.ka.z = 0;
  object.kd.x = 0;
  object.kd.y = 0;
  object.kd.z = 0;
  object.ks.x = 0;
  object.ks.y = 0;
  object.ks.z = 0;
  object.ns = 0;
  object.ni = 1;
  object.nr = 0;
  object.d = 1;
  return object;
}

struct vector3 create_vec(FILE *file)
{
  struct vector3 vec;
  fscanf(file, "%f %f %f", &vec.x, &vec.y, &vec.z);
  return vec;
}

struct triangle create_triangle(struct stack *v, struct stack *vn)
{
  struct triangle triangle;
  for (int i = 0; i < 3; i++)
  {
    triangle.vertex[i] = head_stack(v);
    triangle.normal[i] = head_stack(vn);
    stack_pop(v);
    stack_pop(vn);
  }
  return triangle;
}

struct object parse_object(FILE *file)
{
  struct object object = init_object();
  struct triangle *triangles = (struct triangle *)malloc(sizeof (struct triangle));
  struct stack *v = init_stack();
  struct stack *vn = init_stack();
  (void)fscanf(file, "%u", &object.triangle_count);
  char argument[10];
  unsigned cpt = 0;
  while (cpt < object.triangle_count*2 && fscanf(file, "%s", argument) != EOF)
  {
    if (strcmp(argument, "Ka") == 0)
      object.ka = create_vec(file);
    else if (strcmp(argument, "Kd") == 0)
      object.kd = create_vec(file);
   else if (strcmp(argument, "Ks") == 0)
      object.ks = create_vec(file);

    else if (strcmp(argument, "Ns") == 0)
      fscanf(file, "%f", &object.ns);
    else if (strcmp(argument, "Ni") == 0)
      fscanf(file, "%f", &object.ni);
    else if (strcmp(argument, "Nr") == 0)
      fscanf(file, "%f", &object.nr);
    else if (strcmp(argument, "d") == 0)
      fscanf(file, "%f", &object.d);

    else if (strcmp(argument, "v") == 0)
    {
      cpt++;
      add_stack(v, create_vec(file));
    }
    else if (strcmp(argument, "vn") == 0)
    {
      cpt++;
      add_stack(vn, create_vec(file));
    }
    else
      errx(1, "Error during parsing %s\n", argument);
  }
  cpt = 0;
  while (v->size > 0)
  {
    cpt++;
    triangles = (struct triangle *)realloc(triangles, sizeof (struct triangle) * cpt);
    triangles[cpt-1] = create_triangle(v, vn);
  }
  object.triangle_count /= 3;
  object.triangles = triangles;
  return object;
}
