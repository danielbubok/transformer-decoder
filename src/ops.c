#include "ops.h"
#include <math.h>

void matmul(float *C, float *A, float *B, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int p = 0; p < k; p++) {
        sum += A[i * k + p] * B[p * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

void matmul_transB(float *C, float *A, float *B, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int p = 0; p < k; p++) {
        sum += A[i * k + p] * B[j * k + p];
      }
      C[i * n + j] = sum;
    }
  }
}

void relu_inplace(float *x, int n) {
  for (int i = 0; i < n; i++) {
    if (x[i] < 0.0f)
      x[i] = 0.0f;
  }
}

void softmax(float *x, int n) {
  float max_val = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > max_val)
      max_val = x[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  float inv_sum = 1.0f / sum;
  for (int i = 0; i < n; i++) {
    x[i] *= inv_sum;
  }
}
