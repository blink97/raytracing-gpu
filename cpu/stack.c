#include "stack.h"

struct list *init_list(struct vector3 data)
{
  struct list *list = malloc(sizeof (struct list));
  if (!list)
    errx(1, "%s\n", strerror(errno));
  list->data = data;
  list->next = NULL;
  return list;
}

struct stack *init_stack(void)
{
  struct stack *stack = malloc(sizeof (struct stack));
  if (!stack)
    errx(1, "%s\n", strerror(errno));
  stack->head = NULL;
  stack->size = 0;
  return stack;
}

void add_stack(struct stack *stack, struct vector3 elt)
{
  struct list *new_elt = init_list(elt);
  if (new_elt)
  {
    new_elt->next = stack->head;
    stack->head = new_elt;
    stack->size += 1;
  }
}

struct vector3 head_stack(struct stack *stack)
{
  return stack->head->data;
}

void stack_pop(struct stack *stack)
{
  struct list *tmp = stack->head;
  if (tmp)
  {
    stack->head = tmp->next;
    stack->size -= 1;
  }
}
