#ifndef THREAD_ARG_H
# define THREAD_ARG_H

# include "vector3.h"
# include "scene.h"
# include "colors.h"

struct thread_arg {
  int startx;
  int stopx;
  int starty;
  int stopy;
  int halfh;
  int halfw;
  struct scene scene;
  struct color *output_tab;
  vector3 v;
  vector3 u;
  vector3 C;
};

struct thread_arg init_thread_arg(struct color *output_tab, vector3 v,
                                  vector3 u, vector3 C);

#endif /* !THREAD_ARG_H */
