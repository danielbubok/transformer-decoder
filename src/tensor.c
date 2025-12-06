#include "tensor.h"
#include <math.h>
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

void *safe_realloc(void *ptr, size_t size) {
  void *new_ptr = realloc(ptr, size);
  if (!new_ptr && size > 0) {
    fprintf(stderr, "reallocation failed: %zu bytes\n", size);
    exit(1);
  }
  return new_ptr;
}

// xoshiro256++ prng
uint64_t rng_state[4] = {0x12345678deadbeefULL, 0xfedcba9876543210ULL,
                         0xabcdef0123456789ULL, 0x9876543210abcdefULL};

static inline uint64_t rotl(uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(void) {
  uint64_t result = rotl(rng_state[0] + rng_state[3], 23) + rng_state[0];
  uint64_t t = rng_state[1] << 17;
  rng_state[2] ^= rng_state[0];
  rng_state[3] ^= rng_state[1];
  rng_state[1] ^= rng_state[2];
  rng_state[0] ^= rng_state[3];
  rng_state[2] ^= t;
  rng_state[3] = rotl(rng_state[3], 45);
  return result;
}

void rng_seed(uint64_t seed) {
  rng_state[0] = seed;
  rng_state[1] = seed ^ 0xfedcba9876543210ULL;
  rng_state[2] = seed ^ 0xabcdef0123456789ULL;
  rng_state[3] = seed ^ 0x9876543210abcdefULL;
  // warmup
  for (int i = 0; i < 16; i++)
    rng_next();
}

float rng_uniform(void) {
  return (float)(rng_next() >> 11) * (1.0f / 9007199254740992.0f);
}

float rng_normal(void) {
  // box-muller transform
  float u1 = rng_uniform();
  float u2 = rng_uniform();
  if (u1 < 1e-10f)
    u1 = 1e-10f;
  return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
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
  t->grad = NULL;
  return t;
}

Tensor *tensor_create_2d(int d0, int d1) {
  int shape[] = {d0, d1};
  return tensor_create(2, shape);
}

Tensor *tensor_create_3d(int d0, int d1, int d2) {
  int shape[] = {d0, d1, d2};
  return tensor_create(3, shape);
}

Tensor *tensor_clone(Tensor *src) {
  Tensor *dst = tensor_create(src->ndim, src->shape);
  memcpy(dst->data, src->data, (size_t)src->size * sizeof(float));
  if (src->grad) {
    tensor_alloc_grad(dst);
    memcpy(dst->grad, src->grad, (size_t)src->size * sizeof(float));
  }
  return dst;
}

void tensor_free(Tensor *t) {
  if (t) {
    free(t->data);
    free(t->grad);
    free(t);
  }
}

float *tensor_at_2d(Tensor *t, int i, int j) {
  return &t->data[i * t->shape[1] + j];
}

float *tensor_at_3d(Tensor *t, int i, int j, int k) {
  return &t->data[(i * t->shape[1] + j) * t->shape[2] + k];
}

void tensor_copy(Tensor *dst, Tensor *src) {
  int copy_size = src->size < dst->size ? src->size : dst->size;
  memcpy(dst->data, src->data, (size_t)copy_size * sizeof(float));
}

void tensor_zero(Tensor *t) {
  memset(t->data, 0, (size_t)t->size * sizeof(float));
}

void tensor_zero_grad(Tensor *t) {
  if (t->grad) {
    memset(t->grad, 0, (size_t)t->size * sizeof(float));
  }
}

void tensor_add(Tensor *dst, Tensor *a, Tensor *b) {
  for (int i = 0; i < dst->size; i++) {
    dst->data[i] = a->data[i] + b->data[i];
  }
}

void tensor_add_inplace(Tensor *dst, Tensor *src) {
  for (int i = 0; i < dst->size; i++) {
    dst->data[i] += src->data[i];
  }
}

void tensor_scale(Tensor *t, float s) {
  for (int i = 0; i < t->size; i++) {
    t->data[i] *= s;
  }
}

void tensor_add_scaled(Tensor *dst, Tensor *src, float scale) {
  for (int i = 0; i < dst->size; i++) {
    dst->data[i] += scale * src->data[i];
  }
}

void tensor_init_xavier(Tensor *t, int fan_in, int fan_out) {
  float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
  for (int i = 0; i < t->size; i++) {
    t->data[i] = (2.0f * rng_uniform() - 1.0f) * limit;
  }
}

void tensor_init_zeros(Tensor *t) { tensor_zero(t); }

void tensor_init_ones(Tensor *t) {
  for (int i = 0; i < t->size; i++) {
    t->data[i] = 1.0f;
  }
}

void tensor_alloc_grad(Tensor *t) {
  if (!t->grad) {
    t->grad = (float *)safe_calloc((size_t)t->size, sizeof(float));
  }
}

void tensor_accum_grad(Tensor *t, float *grad) {
  if (!t->grad) {
    tensor_alloc_grad(t);
  }
  for (int i = 0; i < t->size; i++) {
    t->grad[i] += grad[i];
  }
}
