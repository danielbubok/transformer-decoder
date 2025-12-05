CC := gcc
CFLAGS := -I./include -std=c23 -Wall -Wextra -Werror -pedantic -fstrict-aliasing -fno-common
LDFLAGS := -lm

SRC := $(shell find src -type f -name '*.c')
OBJ := $(patsubst src/%.c,obj/%.o,$(SRC))

bin/main: $(OBJ)
	@mkdir -p bin
	$(CC) $(OBJ) -o bin/main $(LDFLAGS)

obj/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf obj bin
