#include "thread_arg.h"

struct thread_arg init_thread_arg(struct color *output_tab, struct vector3 v,
                                  struct vector3 u, struct vector3 C)
{
  struct thread_arg arg;
  arg.output_tab = output_tab;
  arg.v = v;
  arg.u = u;
  arg.C = C;
  return arg;
}
