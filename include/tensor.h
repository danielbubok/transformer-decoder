#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
  float *data;
  int ndim;
  int shape[4];
  int size;
} Tensor;

void *safe_malloc(size_t size);
void *safe_calloc(size_t count, size_t size);

Tensor *tensor_create(int ndim, int *shape);
Tensor *tensor_create_2d(int d0, int d1);
void tensor_free(Tensor *t);
float *tensor_at_2d(Tensor *t, int i, int j);
void tensor_copy(Tensor *dst, Tensor *src);
void tensor_zero(Tensor *t);
void tensor_init_xavier(Tensor *t, int fan_in, int fan_out);

#endif
