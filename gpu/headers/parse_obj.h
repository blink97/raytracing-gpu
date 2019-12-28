#ifndef PARSE_OBJ_H
# define PARSE_OBJ_H

# include <cstdio>

# include "scene.h"

vector3 create_vec(FILE *file);
void parse_object(FILE *file, struct scene *scene);

#endif /* !PARSE_OBJ_H */
