#include "tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *safe_malloc(size_t size) {
  void *ptr = malloc(size);
  if (!ptr && size > 0) {
    fprintf(stderr, "allocation failed: %zu bytes\n", size);
    exit(1);
  }
  return ptr;
}

void *safe_calloc(size_t count, size_t size) {
  void *ptr = calloc(count, size);
  if (!ptr && count > 0 && size > 0) {
    fprintf(stderr, "allocation failed: %zu elements of %zu bytes\n", count,
            size);
    exit(1);
  }
  return ptr;
}

Tensor *tensor_create(int ndim, int *shape) {
  Tensor *t = (Tensor *)safe_malloc(sizeof(Tensor));
  t->ndim = ndim;
  t->size = 1;
  for (int i = 0; i < ndim; i++) {
    t->shape[i] = shape[i];
    t->size *= shape[i];
  }
  for (int i = ndim; i < 4; i++) {
    t->shape[i] = 1;
  }
  t->data = (float *)safe_calloc((size_t)t->size, sizeof(float));
  return t;
}

Tensor *tensor_create_2d(int d0, int d1) {
  int shape[] = {d0, d1};
  return tensor_create(2, shape);
}

void tensor_free(Tensor *t) {
  if (t) {
    free(t->data);
    free(t);
  }
}

float *tensor_at_2d(Tensor *t, int i, int j) {
  return &t->data[i * t->shape[1] + j];
}

void tensor_copy(Tensor *dst, Tensor *src) {
  memcpy(dst->data, src->data, (size_t)src->size * sizeof(float));
}

void tensor_zero(Tensor *t) {
  memset(t->data, 0, (size_t)t->size * sizeof(float));
}

static uint64_t rng_state[2] = {0x12345678deadbeefULL, 0xfedcba9876543210ULL};

static uint64_t rng_next(void) {
  uint64_t s1 = rng_state[0];
  uint64_t s0 = rng_state[1];
  uint64_t result = s0 + s1;
  rng_state[0] = s0;
  s1 ^= s1 << 23;
  rng_state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
  return result;
}

static float rng_uniform(void) {
  return (float)(rng_next() >> 11) * (1.0f / 9007199254740992.0f);
}

static float rng_xavier(int fan_in, int fan_out) {
  float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
  return (2.0f * rng_uniform() - 1.0f) * limit;
}

void tensor_init_xavier(Tensor *t, int fan_in, int fan_out) {
  for (int i = 0; i < t->size; i++) {
    t->data[i] = rng_xavier(fan_in, fan_out);
  }
}
