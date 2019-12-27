#ifndef PARSE_OBJ_H
# define PARSE_OBJ_H

# include "stack.h"
# include "scene.h"

struct object init_object(void);
struct vector3 create_vec(FILE *file);
struct triangle create_triangle(struct stack *v, struct stack *u);
struct object parse_object(FILE *file);

#endif /* !PARSE_OBJ_H */
