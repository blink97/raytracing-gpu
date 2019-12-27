#ifndef PRINTER_H
# define PRINTER_H

# include <err.h>
# include <errno.h>
# include <string.h>
# include <stdio.h>

# include "colors.h"

FILE *open_output(const char *output, int width, int height);
void print_color(struct color color, FILE *output);

#endif /* !PRINT_H */
