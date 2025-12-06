// opencl kernels for moe transformer
// all kernels use float32 precision

// tiled gemm: C[m,n] = A[m,k] * B[k,n]
// tile size 16x16 for coalesced memory access
#define TILE_SIZE 16

__kernel void tiled_gemm(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int K,
    const int N
) {
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int global_row = get_group_id(0) * TILE_SIZE + row;
    const int global_col = get_group_id(1) * TILE_SIZE + col;
    
    __local float A_tile[TILE_SIZE][TILE_SIZE];
    __local float B_tile[TILE_SIZE][TILE_SIZE];
    
    float acc = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        const int tile_k = t * TILE_SIZE;
        
        if (global_row < M && tile_k + col < K) {
            A_tile[row][col] = A[global_row * K + tile_k + col];
        } else {
            A_tile[row][col] = 0.0f;
        }
        
        if (tile_k + row < K && global_col < N) {
            B_tile[row][col] = B[(tile_k + row) * N + global_col];
        } else {
            B_tile[row][col] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += A_tile[row][k] * B_tile[k][col];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = acc;
    }
}

// gemm with transposed B: C[m,n] = A[m,k] * B^T[n,k]
__kernel void tiled_gemm_transB(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int K,
    const int N
) {
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int global_row = get_group_id(0) * TILE_SIZE + row;
    const int global_col = get_group_id(1) * TILE_SIZE + col;
    
    __local float A_tile[TILE_SIZE][TILE_SIZE];
    __local float B_tile[TILE_SIZE][TILE_SIZE];
    
    float acc = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        const int tile_k = t * TILE_SIZE;
        
        if (global_row < M && tile_k + col < K) {
            A_tile[row][col] = A[global_row * K + tile_k + col];
        } else {
            A_tile[row][col] = 0.0f;
        }
        
        if (global_col < N && tile_k + row < K) {
            B_tile[row][col] = B[global_col * K + tile_k + row];
        } else {
            B_tile[row][col] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += A_tile[row][k] * B_tile[k][col];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = acc;
    }
}

// numerically stable softmax with max subtraction
// operates on rows of size N, with stride between rows
__kernel void softmax_stable(
    __global float* x,
    const int rows,
    const int cols
) {
    const int row = get_global_id(0);
    if (row >= rows) return;
    
    __global float* row_data = x + row * cols;
    
    // find max
    float max_val = row_data[0];
    for (int i = 1; i < cols; i++) {
        max_val = fmax(max_val, row_data[i]);
    }
    
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        row_data[i] = exp(row_data[i] - max_val);
        sum += row_data[i];
    }
    
    // normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < cols; i++) {
        row_data[i] *= inv_sum;
    }
}

// rmsnorm: y = x * rsqrt(mean(x^2) + eps) * gamma
__kernel void rmsnorm_forward(
    __global const float* x,
    __global const float* gamma,
    __global float* y,
    const int rows,
    const int dim,
    const float eps
) {
    const int row = get_global_id(0);
    if (row >= rows) return;
    
    __global const float* x_row = x + row * dim;
    __global float* y_row = y + row * dim;
    
    // compute mean of squares
    float ms = 0.0f;
    for (int i = 0; i < dim; i++) {
        ms += x_row[i] * x_row[i];
    }
    ms = ms / (float)dim;
    
    float scale = rsqrt(ms + eps);
    
    for (int i = 0; i < dim; i++) {
        y_row[i] = x_row[i] * scale * gamma[i];
    }
}

// rmsnorm backward
__kernel void rmsnorm_backward(
    __global const float* dy,
    __global const float* x,
    __global const float* gamma,
    __global float* dx,
    __global float* dgamma,
    const int rows,
    const int dim,
    const float eps
) {
    const int row = get_global_id(0);
    if (row >= rows) return;
    
    __global const float* dy_row = dy + row * dim;
    __global const float* x_row = x + row * dim;
    __global float* dx_row = dx + row * dim;
    
    // recompute rms
    float ms = 0.0f;
    for (int i = 0; i < dim; i++) {
        ms += x_row[i] * x_row[i];
    }
    ms = ms / (float)dim;
    float rms_inv = rsqrt(ms + eps);
    
    // dy/dx = gamma * rms_inv * (1 - x^2 / (dim * (ms + eps)))
    float coeff = 0.0f;
    for (int i = 0; i < dim; i++) {
        coeff += dy_row[i] * gamma[i] * x_row[i];
    }
    coeff = coeff * rms_inv * rms_inv * rms_inv / (float)dim;
    
    for (int i = 0; i < dim; i++) {
        dx_row[i] = gamma[i] * rms_inv * dy_row[i] - coeff * x_row[i];
        // atomic add to dgamma
        // dgamma[i] += dy_row[i] * x_row[i] * rms_inv;
    }
}

// rope: apply rotary position embedding
// x has shape [seq_len, dim], dim must be even
// theta_i = pos * 10000^(-2i/dim)
__kernel void rope_forward(
    __global float* x,
    const int seq_len,
    const int dim,
    const float base
) {
    const int pos = get_global_id(0);
    const int i = get_global_id(1);
    
    if (pos >= seq_len || i >= dim / 2) return;
    
    float freq = 1.0f / pow(base, (float)(2 * i) / (float)dim);
    float theta = (float)pos * freq;
    float cos_t = cos(theta);
    float sin_t = sin(theta);
    
    __global float* row = x + pos * dim;
    float x0 = row[i];
    float x1 = row[i + dim / 2];
    
    row[i] = x0 * cos_t - x1 * sin_t;
    row[i + dim / 2] = x0 * sin_t + x1 * cos_t;
}

// swiglu: out = silu(gate) * value
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
__kernel void swiglu_forward(
    __global const float* gate,
    __global const float* value,
    __global float* out,
    const int size
) {
    const int i = get_global_id(0);
    if (i >= size) return;
    
    float g = gate[i];
    float silu_g = g / (1.0f + exp(-g));
    out[i] = silu_g * value[i];
}

// swiglu backward
__kernel void swiglu_backward(
    __global const float* dout,
    __global const float* gate,
    __global const float* value,
    __global float* dgate,
    __global float* dvalue,
    const int size
) {
    const int i = get_global_id(0);
    if (i >= size) return;
    
    float g = gate[i];
    float v = value[i];
    float sigmoid_g = 1.0f / (1.0f + exp(-g));
    float silu_g = g * sigmoid_g;
    float dsilu_g = sigmoid_g * (1.0f + g * (1.0f - sigmoid_g));
    
    float dy = dout[i];
    dgate[i] = dy * v * dsilu_g;
    dvalue[i] = dy * silu_g;
}

// top-k routing: find top-k experts for each token
// logits: [seq_len, n_experts]
// indices: [seq_len, top_k] output
// weights: [seq_len, top_k] output (softmaxed)
__kernel void topk_routing(
    __global const float* logits,
    __global int* indices,
    __global float* weights,
    const int seq_len,
    const int n_experts,
    const int top_k
) {
    const int t = get_global_id(0);
    if (t >= seq_len) return;
    
    __global const float* row = logits + t * n_experts;
    __global int* row_idx = indices + t * top_k;
    __global float* row_w = weights + t * top_k;
    
    // initialize with -inf
    float top_vals[16];  // max top_k = 16
    int top_idxs[16];
    for (int k = 0; k < top_k; k++) {
        top_vals[k] = -1e30f;
        top_idxs[k] = -1;
    }
    
    // find top-k
    for (int e = 0; e < n_experts; e++) {
        float v = row[e];
        // insert into sorted list
        for (int k = 0; k < top_k; k++) {
            if (v > top_vals[k]) {
                // shift down
                for (int j = top_k - 1; j > k; j--) {
                    top_vals[j] = top_vals[j-1];
                    top_idxs[j] = top_idxs[j-1];
                }
                top_vals[k] = v;
                top_idxs[k] = e;
                break;
            }
        }
    }
    
    // softmax over top-k only
    float max_val = top_vals[0];
    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        top_vals[k] = exp(top_vals[k] - max_val);
        sum += top_vals[k];
    }
    float inv_sum = 1.0f / sum;
    
    for (int k = 0; k < top_k; k++) {
        row_idx[k] = top_idxs[k];
        row_w[k] = top_vals[k] * inv_sum;
    }
}

// scaled dot product attention for a single head
// Q: [seq_q, dk], K: [seq_k, dk], V: [seq_k, dv], out: [seq_q, dv]
__kernel void head_attention(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* out,
    __global const int* mask,
    const int seq_q,
    const int seq_k,
    const int dk,
    const int dv,
    const float scale
) {
    const int q_pos = get_global_id(0);
    if (q_pos >= seq_q) return;
    
    // compute attention scores for this query position
    float scores[1024];  // max seq_k
    float max_score = -1e30f;
    
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        float score = 0.0f;
        for (int d = 0; d < dk; d++) {
            score += Q[q_pos * dk + d] * K[k_pos * dk + d];
        }
        score *= scale;
        
        // apply mask
        if (mask != 0 && mask[q_pos * seq_k + k_pos] == 0) {
            score = -1e30f;
        }
        
        scores[k_pos] = score;
        max_score = fmax(max_score, score);
    }
    
    // softmax
    float sum = 0.0f;
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        scores[k_pos] = exp(scores[k_pos] - max_score);
        sum += scores[k_pos];
    }
    float inv_sum = 1.0f / sum;
    
    // weighted sum of values
    for (int d = 0; d < dv; d++) {
        float acc = 0.0f;
        for (int k_pos = 0; k_pos < seq_k; k_pos++) {
            acc += scores[k_pos] * inv_sum * V[k_pos * dv + d];
        }
        out[q_pos * dv + d] = acc;
    }
}

// elementwise operations
__kernel void elementwise_add(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int size
) {
    const int i = get_global_id(0);
    if (i >= size) return;
    c[i] = a[i] + b[i];
}

__kernel void elementwise_mul(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int size
) {
    const int i = get_global_id(0);
    if (i >= size) return;
    c[i] = a[i] * b[i];
}

__kernel void elementwise_scale(
    __global float* x,
    const float scale,
    const int size
) {
    const int i = get_global_id(0);
    if (i >= size) return;
    x[i] *= scale;
}

__kernel void elementwise_add_scaled(
    __global float* dst,
    __global const float* src,
    const float scale,
    const int size
) {
    const int i = get_global_id(0);
    if (i >= size) return;
    dst[i] += scale * src[i];
}

// bias add: y[i,j] += bias[j]
__kernel void bias_add(
    __global float* y,
    __global const float* bias,
    const int rows,
    const int cols
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    if (row >= rows || col >= cols) return;
    y[row * cols + col] += bias[col];
}

// cross entropy loss forward
// logits: [seq_len, vocab_size], targets: [seq_len]
// returns per-token losses in out
__kernel void cross_entropy_forward(
    __global const float* logits,
    __global const int* targets,
    __global float* losses,
    const int seq_len,
    const int vocab_size
) {
    const int t = get_global_id(0);
    if (t >= seq_len) return;
    
    __global const float* row = logits + t * vocab_size;
    int target = targets[t];
    
    // find max for numerical stability
    float max_val = row[0];
    for (int i = 1; i < vocab_size; i++) {
        max_val = fmax(max_val, row[i]);
    }
    
    // compute log-sum-exp
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        sum += exp(row[i] - max_val);
    }
    float log_sum = max_val + log(sum);
    
    losses[t] = log_sum - row[target];
}

// cross entropy backward
// writes softmax(logits) - one_hot(target) to dlogits
__kernel void cross_entropy_backward(
    __global const float* logits,
    __global const int* targets,
    __global float* dlogits,
    const int seq_len,
    const int vocab_size
) {
    const int t = get_global_id(0);
    if (t >= seq_len) return;
    
    __global const float* row = logits + t * vocab_size;
    __global float* drow = dlogits + t * vocab_size;
    int target = targets[t];
    
    // compute softmax
    float max_val = row[0];
    for (int i = 1; i < vocab_size; i++) {
        max_val = fmax(max_val, row[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        drow[i] = exp(row[i] - max_val);
        sum += drow[i];
    }
    
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++) {
        drow[i] *= inv_sum;
    }
    drow[target] -= 1.0f;
    
    // scale by 1/seq_len for mean
    float scale = 1.0f / (float)seq_len;
    for (int i = 0; i < vocab_size; i++) {
        drow[i] *= scale;
    }
}

// gradient clipping: compute global L2 norm
__kernel void compute_gradient_norm_partial(
    __global const float* grads,
    __global float* partial_sums,
    const int size,
    const int work_per_item
) {
    const int id = get_global_id(0);
    const int start = id * work_per_item;
    const int end = min(start + work_per_item, size);
    
    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        sum += grads[i] * grads[i];
    }
    partial_sums[id] = sum;
}

// adam update step
__kernel void adam_update_kernel(
    __global float* params,
    __global const float* grads,
    __global float* m,
    __global float* v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bc1,
    const float bc2,
    const int size
) {
    const int i = get_global_id(0);
    if (i >= size) return;
    
    float g = grads[i] + weight_decay * params[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
    
    float m_hat = m[i] / bc1;
    float v_hat = v[i] / bc2;
    
    params[i] -= lr * m_hat / (sqrt(v_hat) + eps);
}

// === FP16 KERNELS ===
// requires cl_khr_fp16 extension

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// tiled gemm for fp16: C[m,n] = A[m,k] * B[k,n]
// computes in fp16, accumulates in fp32 for precision
__kernel void tiled_gemm_f16(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N
) {
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int global_row = get_group_id(0) * TILE_SIZE + row;
    const int global_col = get_group_id(1) * TILE_SIZE + col;
    
    __local float A_tile[TILE_SIZE][TILE_SIZE];
    __local float B_tile[TILE_SIZE][TILE_SIZE];
    
    float acc = 0.0f;
    
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE + col;
        int b_row = t * TILE_SIZE + row;
        
        if (global_row < M && a_col < K) {
            A_tile[row][col] = vload_half(global_row * K + a_col, A);
        } else {
            A_tile[row][col] = 0.0f;
        }
        
        if (b_row < K && global_col < N) {
            B_tile[row][col] = vload_half(b_row * N + global_col, B);
        } else {
            B_tile[row][col] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int i = 0; i < TILE_SIZE; i++) {
            acc += A_tile[row][i] * B_tile[i][col];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (global_row < M && global_col < N) {
        vstore_half(acc, global_row * N + global_col, C);
    }
}

// softmax for fp16
__kernel void softmax_stable_f16(
    __global half* x,
    const int rows,
    const int cols
) {
    const int row = get_global_id(0);
    if (row >= rows) return;
    
    __global half* row_data = x + row * cols;
    
    // find max in fp32
    float max_val = vload_half(0, row_data);
    for (int i = 1; i < cols; i++) {
        float val = vload_half(i, row_data);
        if (val > max_val) max_val = val;
    }
    
    // compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        float val = vload_half(i, row_data);
        float exp_val = exp(val - max_val);
        vstore_half(exp_val, i, row_data);
        sum += exp_val;
    }
    
    // normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < cols; i++) {
        float val = vload_half(i, row_data);
        vstore_half(val * inv_sum, i, row_data);
    }
}

// rmsnorm for fp16
__kernel void rmsnorm_forward_f16(
    __global const half* x,
    __global const half* gamma,
    __global half* y,
    const int rows,
    const int dim,
    const float eps
) {
    const int row = get_global_id(0);
    if (row >= rows) return;
    
    __global const half* x_row = x + row * dim;
    __global half* y_row = y + row * dim;
    
    // compute rms in fp32
    float ms = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = vload_half(i, x_row);
        ms += val * val;
    }
    ms = ms / (float)dim;
    
    float scale = rsqrt(ms + eps);
    
    for (int i = 0; i < dim; i++) {
        float x_val = vload_half(i, x_row);
        float g_val = vload_half(i, gamma);
        vstore_half(x_val * scale * g_val, i, y_row);
    }
}

// swiglu for fp16
__kernel void swiglu_forward_f16(
    __global const half* gate,
    __global const half* value,
    __global half* out,
    const int size
) {
    const int i = get_global_id(0);
    if (i >= size) return;
    
    float g = vload_half(i, gate);
    float v = vload_half(i, value);
    
    // silu(g) * v
    float sigmoid_g = 1.0f / (1.0f + exp(-g));
    float silu = g * sigmoid_g;
    vstore_half(silu * v, i, out);
}

// elementwise add for fp16
__kernel void elementwise_add_f16(
    __global const half* a,
    __global const half* b,
    __global half* c,
    const int size
) {
    const int i = get_global_id(0);
    if (i >= size) return;
    float va = vload_half(i, a);
    float vb = vload_half(i, b);
    vstore_half(va + vb, i, c);
}

// scale for fp16
__kernel void elementwise_scale_f16(
    __global half* x,
    const float s,
    const int size
) {
    const int i = get_global_id(0);
    if (i >= size) return;
    float v = vload_half(i, x);
    vstore_half(v * s, i, x);
}
