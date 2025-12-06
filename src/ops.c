#include "ops.h"
#include <math.h>
#include <string.h>

void matmul(float *C, const float *A, const float *B, int m, int k, int n) {
  // C[m,n] = A[m,k] * B[k,n]
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

void matmul_transB(float *C, const float *A, const float *B, int m, int k,
                   int n) {
  // C[m,n] = A[m,k] * B^T[n,k]
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

void matmul_transA(float *C, const float *A, const float *B, int m, int k,
                   int n) {
  // C[m,n] = A^T[m,k] * B[k,n], where A is stored as [k,m]
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int p = 0; p < k; p++) {
        sum += A[p * m + i] * B[p * n + j];
      }
      C[i * n + j] = sum;
    }
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

void softmax_rows(float *x, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    softmax(x + r * cols, cols);
  }
}

static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

void silu(float *x, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = x[i] * sigmoid(x[i]);
  }
}

void silu_backward(float *dx, const float *dy, const float *x, int n) {
  for (int i = 0; i < n; i++) {
    float s = sigmoid(x[i]);
    float silu_grad = s * (1.0f + x[i] * (1.0f - s));
    dx[i] = dy[i] * silu_grad;
  }
}

void swiglu(float *out, const float *gate, const float *value, int n) {
  // out = silu(gate) * value
  for (int i = 0; i < n; i++) {
    float g = gate[i];
    float silu_g = g * sigmoid(g);
    out[i] = silu_g * value[i];
  }
}

void swiglu_backward(float *dgate, float *dvalue, const float *dout,
                     const float *gate, const float *value, int n) {
  for (int i = 0; i < n; i++) {
    float g = gate[i];
    float v = value[i];
    float s = sigmoid(g);
    float silu_g = g * s;
    float dsilu_g = s * (1.0f + g * (1.0f - s));

    dgate[i] = dout[i] * v * dsilu_g;
    dvalue[i] = dout[i] * silu_g;
  }
}

void rmsnorm_cpu(float *out, const float *x, const float *gamma, int rows,
                 int dim, float eps) {
  for (int r = 0; r < rows; r++) {
    const float *x_row = x + r * dim;
    float *out_row = out + r * dim;

    // compute mean of squares
    float ms = 0.0f;
    for (int i = 0; i < dim; i++) {
      ms += x_row[i] * x_row[i];
    }
    ms = ms / (float)dim;

    float scale = 1.0f / sqrtf(ms + eps);

    for (int i = 0; i < dim; i++) {
      out_row[i] = x_row[i] * scale * gamma[i];
    }
  }
}

void rmsnorm_backward_cpu(float *dx, float *dgamma, const float *dy,
                          const float *x, const float *gamma, int rows, int dim,
                          float eps) {
  // zero dgamma accumulator
  memset(dgamma, 0, (size_t)dim * sizeof(float));

  for (int r = 0; r < rows; r++) {
    const float *dy_row = dy + r * dim;
    const float *x_row = x + r * dim;
    float *dx_row = dx + r * dim;

    // recompute rms
    float ms = 0.0f;
    for (int i = 0; i < dim; i++) {
      ms += x_row[i] * x_row[i];
    }
    ms = ms / (float)dim;
    float rms_inv = 1.0f / sqrtf(ms + eps);

    // accumulate dgamma
    for (int i = 0; i < dim; i++) {
      dgamma[i] += dy_row[i] * x_row[i] * rms_inv;
    }

    // compute dx
    // dx = gamma * rms_inv * (dy - x * (sum(dy * gamma * x) / (dim * (ms +
    // eps))))
    float coeff = 0.0f;
    for (int i = 0; i < dim; i++) {
      coeff += dy_row[i] * gamma[i] * x_row[i];
    }
    coeff = coeff * rms_inv * rms_inv * rms_inv / (float)dim;

    for (int i = 0; i < dim; i++) {
      dx_row[i] = gamma[i] * rms_inv * dy_row[i] - coeff * x_row[i];
    }
  }
}

void rope_forward(float *x, int seq_len, int dim, float base) {
  int half = dim / 2;
  for (int pos = 0; pos < seq_len; pos++) {
    float *row = x + pos * dim;
    for (int i = 0; i < half; i++) {
      float freq = 1.0f / powf(base, (float)(2 * i) / (float)dim);
      float theta = (float)pos * freq;
      float cos_t = cosf(theta);
      float sin_t = sinf(theta);

      float x0 = row[i];
      float x1 = row[i + half];
      row[i] = x0 * cos_t - x1 * sin_t;
      row[i + half] = x0 * sin_t + x1 * cos_t;
    }
  }
}

void rope_backward(float *dx, const float *dy, int seq_len, int dim,
                   float base) {
  // rope is orthogonal, so backward is just apply with negated angle
  int half = dim / 2;
  for (int pos = 0; pos < seq_len; pos++) {
    float *dx_row = dx + pos * dim;
    const float *dy_row = dy + pos * dim;
    for (int i = 0; i < half; i++) {
      float freq = 1.0f / powf(base, (float)(2 * i) / (float)dim);
      float theta = (float)pos * freq;
      float cos_t = cosf(theta);
      float sin_t = sinf(theta);

      // inverse rotation (negate sin)
      dx_row[i] = dy_row[i] * cos_t + dy_row[i + half] * sin_t;
      dx_row[i + half] = -dy_row[i] * sin_t + dy_row[i + half] * cos_t;
    }
  }
}

void topk_indices(int *indices, float *weights, const float *logits, int n,
                  int k) {
  // initialize
  for (int i = 0; i < k; i++) {
    indices[i] = -1;
    weights[i] = -1e30f;
  }

  // find top-k
  for (int e = 0; e < n; e++) {
    float v = logits[e];
    for (int i = 0; i < k; i++) {
      if (v > weights[i]) {
        // shift down
        for (int j = k - 1; j > i; j--) {
          weights[j] = weights[j - 1];
          indices[j] = indices[j - 1];
        }
        weights[i] = v;
        indices[i] = e;
        break;
      }
    }
  }

  // softmax over top-k
  float max_val = weights[0];
  float sum = 0.0f;
  for (int i = 0; i < k; i++) {
    weights[i] = expf(weights[i] - max_val);
    sum += weights[i];
  }
  float inv_sum = 1.0f / sum;
  for (int i = 0; i < k; i++) {
    weights[i] *= inv_sum;
  }
}

void add_bias(float *y, const float *bias, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      y[r * cols + c] += bias[c];
    }
  }
}

void add_inplace(float *dst, const float *src, int n) {
  for (int i = 0; i < n; i++) {
    dst[i] += src[i];
  }
}

void scale_inplace(float *x, float s, int n) {
  for (int i = 0; i < n; i++) {
    x[i] *= s;
  }
}

void copy_strided(float *dst, int dst_stride, const float *src, int src_stride,
                  int count, int elem_size) {
  for (int i = 0; i < count; i++) {
    memcpy(dst + i * dst_stride, src + i * src_stride,
           (size_t)elem_size * sizeof(float));
  }
}

float reduce_sum(const float *x, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  return sum;
}

float reduce_max(const float *x, int n) {
  float max_val = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > max_val)
      max_val = x[i];
  }
  return max_val;
}

float l2_norm(const float *x, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += x[i] * x[i];
  }
  return sqrtf(sum);
}
