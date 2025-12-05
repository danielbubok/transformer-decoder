#ifndef OPS_H
#define OPS_H

void matmul(float *C, float *A, float *B, int m, int k, int n);
void matmul_transB(float *C, float *A, float *B, int m, int k, int n);
void relu_inplace(float *x, int n);
void softmax(float *x, int n);

#endif
