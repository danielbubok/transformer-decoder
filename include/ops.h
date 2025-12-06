#ifndef OPS_H
#define OPS_H

#include <stddef.h>

// matrix operations (cpu fallback)
void matmul(float *C, const float *A, const float *B, int m, int k, int n);
void matmul_transB(float *C, const float *A, const float *B, int m, int k,
                   int n);
void matmul_transA(float *C, const float *A, const float *B, int m, int k,
                   int n);

// activations
void softmax(float *x, int n);
void softmax_rows(float *x, int rows, int cols);
void swiglu(float *out, const float *gate, const float *value, int n);
void swiglu_backward(float *dgate, float *dvalue, const float *dout,
                     const float *gate, const float *value, int n);
void silu(float *x, int n);
void silu_backward(float *dx, const float *dy, const float *x, int n);

// normalization
void rmsnorm_cpu(float *out, const float *x, const float *gamma, int rows,
                 int dim, float eps);
void rmsnorm_backward_cpu(float *dx, float *dgamma, const float *dy,
                          const float *x, const float *gamma, int rows, int dim,
                          float eps);

// rope
void rope_forward(float *x, int seq_len, int dim, float base);
void rope_backward(float *dx, const float *dy, int seq_len, int dim,
                   float base);

// top-k
void topk_indices(int *indices, float *weights, const float *logits, int n,
                  int k);

// elementwise
void add_bias(float *y, const float *bias, int rows, int cols);
void add_inplace(float *dst, const float *src, int n);
void scale_inplace(float *x, float s, int n);
void copy_strided(float *dst, int dst_stride, const float *src, int src_stride,
                  int count, int elem_size);

// reduction
float reduce_sum(const float *x, int n);
float reduce_max(const float *x, int n);
float l2_norm(const float *x, int n);

#endif
