#ifndef STACK_H
# define STACK_H

# include <errno.h>
# include <err.h>
# include <stddef.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>

# include "vector3.h"

struct list
{
  struct vector3 data;
  struct list *next;
};

struct stack
{
  struct list *head;
  size_t size;
};

struct list *list_init(struct vector3 data);
struct stack *init_stack(void);
void add_stack(struct stack *stack, struct vector3 elt);
struct vector3 head_stack(struct stack *stack);
void stack_pop(struct stack *stack);

#endif /* !STACK_H */
