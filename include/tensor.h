#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
  float *data;
  float *grad;
  int ndim;
  int shape[4];
  int size;
} Tensor;

// memory allocation
void *safe_malloc(size_t size);
void *safe_calloc(size_t count, size_t size);
void *safe_realloc(void *ptr, size_t size);

// tensor lifecycle
Tensor *tensor_create(int ndim, int *shape);
Tensor *tensor_create_2d(int d0, int d1);
Tensor *tensor_create_3d(int d0, int d1, int d2);
Tensor *tensor_clone(Tensor *src);
void tensor_free(Tensor *t);

// accessors
float *tensor_at_2d(Tensor *t, int i, int j);
float *tensor_at_3d(Tensor *t, int i, int j, int k);

// operations
void tensor_copy(Tensor *dst, Tensor *src);
void tensor_zero(Tensor *t);
void tensor_zero_grad(Tensor *t);
void tensor_add(Tensor *dst, Tensor *a, Tensor *b);
void tensor_add_inplace(Tensor *dst, Tensor *src);
void tensor_scale(Tensor *t, float s);
void tensor_add_scaled(Tensor *dst, Tensor *src, float scale);

// initialization
void tensor_init_xavier(Tensor *t, int fan_in, int fan_out);
void tensor_init_zeros(Tensor *t);
void tensor_init_ones(Tensor *t);

// gradient management
void tensor_alloc_grad(Tensor *t);
void tensor_accum_grad(Tensor *t, float *grad);

// xoshiro256++ prng
extern uint64_t rng_state[4];
void rng_seed(uint64_t seed);
float rng_uniform(void);
float rng_normal(void);

#endif
