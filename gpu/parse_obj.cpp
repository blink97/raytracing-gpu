#include "parse_obj.h"

# include <cstdlib>
# include <cstring>
# include <err.h>
# include <stack>


vector3 create_vec(FILE *file)
{
  vector3 vec;
  fscanf(file, "%f %f %f", &vec.x, &vec.y, &vec.z);
  return vec;
}

struct object parse_object(FILE *file)
{
  struct object object = {
    .triangles = nullptr,
    .triangle_count = 0,
    .ka = { .x = 0, .y = 0, .z = 0 },
    .kd = { .x = 0, .y = 0, .z = 0 },
    .ks = { .x = 0, .y = 0, .z = 0 },
    .ns = 0,
    .ni = 1,
    .nr = 0,
    .d = 1
  };

  (void)fscanf(file, "%u", &object.triangle_count);

  std::stack<vector3> v;
  std::stack<vector3> vn;

  char argument[10];

  while ((v.size() < object.triangle_count || vn.size() < object.triangle_count) &&
         fscanf(file, "%s", argument) != EOF)
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
      v.push(create_vec(file));
    else if (strcmp(argument, "vn") == 0)
      vn.push(create_vec(file));
    else
      errx(1, "Error during parsing %s\n", argument);
  }

  object.triangle_count /= 3;
  object.triangles = (struct triangle *)malloc(sizeof(struct triangle) * v.size());

  for (unsigned int i = 0; i < object.triangle_count; ++i)
  {
    for (int j = 0; j < 3; j++)
    {
      object.triangles[i].vertex[j] = v.top();
      object.triangles[i].normal[j] = vn.top();
      v.pop();
      vn.pop();
    }
  }

  return object;
}
