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

void parse_object(FILE *file, struct scene *scene)
{
  uint32_t vertex_count;
  (void)fscanf(file, "%u", &vertex_count);

  struct object *object = add_object_to_scene(scene, vertex_count / 3);

  std::stack<vector3> v;
  std::stack<vector3> vn;

  char argument[10];

  while ((v.size() < vertex_count || vn.size() < vertex_count/* Same amount of normal as vertex */) &&
         fscanf(file, "%s", argument) != EOF)
  {
    if (strcmp(argument, "Ka") == 0)
      object->ka = create_vec(file);
    else if (strcmp(argument, "Kd") == 0)
      object->kd = create_vec(file);
   else if (strcmp(argument, "Ks") == 0)
      object->ks = create_vec(file);

    else if (strcmp(argument, "Ns") == 0)
      fscanf(file, "%f", &object->ns);
    else if (strcmp(argument, "Ni") == 0)
      fscanf(file, "%f", &object->ni);
    else if (strcmp(argument, "Nr") == 0)
      fscanf(file, "%f", &object->nr);
    else if (strcmp(argument, "d") == 0)
      fscanf(file, "%f", &object->d);

    else if (strcmp(argument, "v") == 0)
      v.push(create_vec(file));
    else if (strcmp(argument, "vn") == 0)
      vn.push(create_vec(file));
    else
      errx(1, "Error during parsing %s\n", argument);
  }

  for (unsigned int i = 0; i < object->triangle_count; ++i)
  {
    for (int j = 0; j < 3; j++)
    {
      get_vertex(object->triangles, i)[j] = v.top(); v.pop();
      get_normal(object->triangles, i)[j] = vn.top(); vn.pop();
    }
  }
}
