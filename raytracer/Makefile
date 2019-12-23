CC = gcc

CPPFLAGS =
DEBUG ?=
CFLAGS = -Wall -Wextra -Werror -std=c99 -pedantic -I./headers ${DEBUG}
LDFLAGS = -lm -pthread
LDLIBS =
VPATH = ./

SRC ?= rt.c raytracer.c parser.c parse_obj.c vector3.c vector3-extern.c hit.c colors.c ray.c stack.c printer.c light.c thread_arg.c
OBJ = ${SRC:.c=.o}
EXEC = rt

all: ${EXEC}

${EXEC}: ${OBJ}

.PHONY: all clean

clean:
	${RM} ${OBJ}
	${RM} ${EXEC}
	${RM} -r *.ppm
