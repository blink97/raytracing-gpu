#ifndef PARSER_H
# define PARSER_H

# include <stdio.h>
# include <string.h>
# include <errno.h>
# include <err.h>

# include "vector3.h"
# include "scene.h"

struct camera parse_camera(FILE *file);
struct light parse_a_light(FILE *file);
struct light parse_d_light(FILE *file);
struct light parse_p_light(FILE *file);


struct scene parser(const char *path);

#endif /* !PARSER_H */
