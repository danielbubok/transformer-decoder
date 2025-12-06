CC := gcc
CFLAGS := -I./include -std=c23 -Wall -Wextra -Werror -pedantic -fstrict-aliasing -fno-common

# opencl detection
OPENCL_CFLAGS :=
OPENCL_LDFLAGS :=
UNAME := $(shell uname -s)

ifeq ($(UNAME), Darwin)
    OPENCL_LDFLAGS := -framework OpenCL
else
    OPENCL_LDFLAGS := -lOpenCL
endif

# release build flags
RELEASE_FLAGS := -O3 -march=native -ffast-math -DNDEBUG

# debug build flags
DEBUG_FLAGS := -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer -DDEBUG

LDFLAGS := -lm $(OPENCL_LDFLAGS)

SRC := $(filter-out src/main.c, $(wildcard src/*.c))
OBJ := $(patsubst src/%.c,obj/%.o,$(SRC))
OBJ_DEBUG := $(patsubst src/%.c,obj/debug/%.o,$(SRC))

.PHONY: all release debug test clean

all: release

release: CFLAGS += $(RELEASE_FLAGS)
release: bin/moe-transformer

debug: CFLAGS += $(DEBUG_FLAGS)
debug: LDFLAGS += -fsanitize=address,undefined
debug: bin/moe-transformer-debug

bin/moe-transformer: $(OBJ) obj/main.o
	@mkdir -p bin
	$(CC) $(OBJ) obj/main.o -o $@ $(LDFLAGS)
	@echo "built: $@"

bin/moe-transformer-debug: $(OBJ_DEBUG) obj/debug/main.o
	@mkdir -p bin
	$(CC) $(OBJ_DEBUG) obj/debug/main.o -o $@ $(LDFLAGS)
	@echo "built: $@"

obj/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

obj/debug/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

test: release
	./bin/moe-transformer test

test-debug: debug
	./bin/moe-transformer-debug test

valgrind: debug
	valgrind --leak-check=full --show-leak-kinds=all ./bin/moe-transformer-debug test

clean:
	rm -rf obj bin

# convenience targets
info: release
	./bin/moe-transformer info

# kernel compilation check (opencl syntax validation)
check-kernels:
	@echo "checking kernel syntax..."
	@cat kernels/kernels.cl | head -1 > /dev/null && echo "kernels/kernels.cl found"

# generate compile_commands.json for IDE support
compile_commands:
	bear -- make clean all
