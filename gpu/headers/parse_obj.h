#ifndef PARSE_OBJ_H
# define PARSE_OBJ_H

# include <cstdio>

# include "scene.h"

vector3 create_vec(FILE *file);
struct object parse_object(FILE *file);

#endif /* !PARSE_OBJ_H */
