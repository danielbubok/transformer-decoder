#include "layers.h"
#include "ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// === RMSNorm ===

RMSNorm *rmsnorm_create(int dim) {
  RMSNorm *rn = (RMSNorm *)safe_malloc(sizeof(RMSNorm));
  rn->dim = dim;
  rn->eps = 1e-6f;
  rn->gamma = tensor_create_2d(1, dim);
  tensor_init_ones(rn->gamma);
  return rn;
}

void rmsnorm_free(RMSNorm *rn) {
  if (rn) {
    tensor_free(rn->gamma);
    free(rn);
  }
}

void rmsnorm_forward(RMSNorm *rn, Tensor *out, Tensor *x, int seq_len) {
  rmsnorm_cpu(out->data, x->data, rn->gamma->data, seq_len, rn->dim, rn->eps);
}

void rmsnorm_backward(RMSNorm *rn, Tensor *dx, Tensor *dy, Tensor *x,
                      int seq_len) {
  float *dgamma = (float *)safe_calloc((size_t)rn->dim, sizeof(float));
  rmsnorm_backward_cpu(dx->data, dgamma, dy->data, x->data, rn->gamma->data,
                       seq_len, rn->dim, rn->eps);
  tensor_accum_grad(rn->gamma, dgamma);
  free(dgamma);
}

// === Linear ===

Linear *linear_create(int in_dim, int out_dim, int use_bias) {
  Linear *l = (Linear *)safe_malloc(sizeof(Linear));
  l->in_dim = in_dim;
  l->out_dim = out_dim;
  l->use_bias = use_bias;
  l->weight = tensor_create_2d(in_dim, out_dim);
  tensor_init_xavier(l->weight, in_dim, out_dim);
  if (use_bias) {
    l->bias = tensor_create_2d(1, out_dim);
    tensor_zero(l->bias);
  } else {
    l->bias = NULL;
  }
  return l;
}

void linear_free(Linear *l) {
  if (l) {
    tensor_free(l->weight);
    tensor_free(l->bias);
    free(l);
  }
}

void linear_forward(Linear *l, Tensor *y, Tensor *x, int seq_len) {
  matmul(y->data, x->data, l->weight->data, seq_len, l->in_dim, l->out_dim);
  if (l->bias) {
    add_bias(y->data, l->bias->data, seq_len, l->out_dim);
  }
}

void linear_backward(Linear *l, Tensor *dx, Tensor *dy, Tensor *x,
                     int seq_len) {
  // grad w.r.t input: dx = dy @ W^T
  matmul_transB(dx->data, dy->data, l->weight->data, seq_len, l->out_dim,
                l->in_dim);

  // grad w.r.t weight: dW = x^T @ dy
  float *dw =
      (float *)safe_calloc((size_t)(l->in_dim * l->out_dim), sizeof(float));
  matmul_transA(dw, x->data, dy->data, l->in_dim, seq_len, l->out_dim);
  tensor_accum_grad(l->weight, dw);
  free(dw);

  // grad w.r.t bias: db = sum(dy, axis=0)
  if (l->bias) {
    float *db = (float *)safe_calloc((size_t)l->out_dim, sizeof(float));
    for (int t = 0; t < seq_len; t++) {
      for (int i = 0; i < l->out_dim; i++) {
        db[i] += dy->data[t * l->out_dim + i];
      }
    }
    tensor_accum_grad(l->bias, db);
    free(db);
  }
}

// === SwiGLU FFN ===

SwiGLUFFN *swiglu_ffn_create(int d_model, int d_ff, int max_seq_len) {
  SwiGLUFFN *ff = (SwiGLUFFN *)safe_malloc(sizeof(SwiGLUFFN));
  ff->d_model = d_model;
  ff->d_ff = d_ff;
  ff->w_gate = linear_create(d_model, d_ff, 0);
  ff->w_value = linear_create(d_model, d_ff, 0);
  ff->w_down = linear_create(d_ff, d_model, 0);
  ff->gate_out = tensor_create_2d(max_seq_len, d_ff);
  ff->value_out = tensor_create_2d(max_seq_len, d_ff);
  ff->swiglu_out = tensor_create_2d(max_seq_len, d_ff);
  return ff;
}

void swiglu_ffn_free(SwiGLUFFN *ff) {
  if (ff) {
    linear_free(ff->w_gate);
    linear_free(ff->w_value);
    linear_free(ff->w_down);
    tensor_free(ff->gate_out);
    tensor_free(ff->value_out);
    tensor_free(ff->swiglu_out);
    free(ff);
  }
}

void swiglu_ffn_forward(SwiGLUFFN *ff, Tensor *out, Tensor *x, int seq_len) {
  linear_forward(ff->w_gate, ff->gate_out, x, seq_len);
  linear_forward(ff->w_value, ff->value_out, x, seq_len);
  swiglu(ff->swiglu_out->data, ff->gate_out->data, ff->value_out->data,
         seq_len * ff->d_ff);
  linear_forward(ff->w_down, out, ff->swiglu_out, seq_len);
}

void swiglu_ffn_backward(SwiGLUFFN *ff, Tensor *dx, Tensor *dy, Tensor *x,
                         int seq_len) {
  int size = seq_len * ff->d_ff;

  // backward through down projection
  Tensor *d_swiglu = tensor_create_2d(seq_len, ff->d_ff);
  linear_backward(ff->w_down, d_swiglu, dy, ff->swiglu_out, seq_len);

  // backward through swiglu
  Tensor *d_gate = tensor_create_2d(seq_len, ff->d_ff);
  Tensor *d_value = tensor_create_2d(seq_len, ff->d_ff);
  swiglu_backward(d_gate->data, d_value->data, d_swiglu->data,
                  ff->gate_out->data, ff->value_out->data, size);

  // backward through gate and value projections
  Tensor *dx_gate = tensor_create_2d(seq_len, ff->d_model);
  Tensor *dx_value = tensor_create_2d(seq_len, ff->d_model);
  linear_backward(ff->w_gate, dx_gate, d_gate, x, seq_len);
  linear_backward(ff->w_value, dx_value, d_value, x, seq_len);

  // combine gradients
  for (int i = 0; i < seq_len * ff->d_model; i++) {
    dx->data[i] = dx_gate->data[i] + dx_value->data[i];
  }

  tensor_free(d_swiglu);
  tensor_free(d_gate);
  tensor_free(d_value);
  tensor_free(dx_gate);
  tensor_free(dx_value);
}

// === RoPE ===

RoPE *rope_create(int max_len, int dim, float base) {
  RoPE *rope = (RoPE *)safe_malloc(sizeof(RoPE));
  rope->max_len = max_len;
  rope->dim = dim;
  rope->base = base;
  rope->sin_cache = tensor_create_2d(max_len, dim);
  rope->cos_cache = tensor_create_2d(max_len, dim);

  int half = dim / 2;
  for (int pos = 0; pos < max_len; pos++) {
    for (int i = 0; i < half; i++) {
      float freq = 1.0f / powf(base, (float)(2 * i) / (float)dim);
      float theta = (float)pos * freq;
      float s = sinf(theta);
      float c = cosf(theta);
      *tensor_at_2d(rope->sin_cache, pos, i) = s;
      *tensor_at_2d(rope->sin_cache, pos, i + half) = s;
      *tensor_at_2d(rope->cos_cache, pos, i) = c;
      *tensor_at_2d(rope->cos_cache, pos, i + half) = c;
    }
  }
  return rope;
}

void rope_free(RoPE *rope) {
  if (rope) {
    tensor_free(rope->sin_cache);
    tensor_free(rope->cos_cache);
    free(rope);
  }
}

void rope_apply(RoPE *rope, Tensor *x, int seq_len) {
  rope_forward(x->data, seq_len, rope->dim, rope->base);
}

// === MLA ===

MLA *mla_create(int n_heads, int d_model, int dk, int dv, int d_kv_comp,
                int max_seq_len) {
  MLA *mla = (MLA *)safe_malloc(sizeof(MLA));
  mla->n_heads = n_heads;
  mla->d_model = d_model;
  mla->dk = dk;
  mla->dv = dv;
  mla->d_kv_comp = d_kv_comp;
  mla->scale = 1.0f / sqrtf((float)dk);

  // projections
  mla->w_q = linear_create(d_model, n_heads * dk, 0);
  mla->w_kv_down = linear_create(d_model, d_kv_comp, 0);

  // dynamically allocate per-head arrays
  mla->w_k_up = (Linear **)safe_malloc((size_t)n_heads * sizeof(Linear *));
  mla->w_v_up = (Linear **)safe_malloc((size_t)n_heads * sizeof(Linear *));
  mla->k_proj = (Tensor **)safe_malloc((size_t)n_heads * sizeof(Tensor *));
  mla->v_proj = (Tensor **)safe_malloc((size_t)n_heads * sizeof(Tensor *));
  mla->head_out = (Tensor **)safe_malloc((size_t)n_heads * sizeof(Tensor *));

  for (int h = 0; h < n_heads; h++) {
    mla->w_k_up[h] = linear_create(d_kv_comp, dk, 0);
    mla->w_v_up[h] = linear_create(d_kv_comp, dv, 0);
    mla->k_proj[h] = tensor_create_2d(max_seq_len, dk);
    mla->v_proj[h] = tensor_create_2d(max_seq_len, dv);
    mla->head_out[h] = tensor_create_2d(max_seq_len, dv);
  }
  mla->w_o = linear_create(n_heads * dv, d_model, 0);

  // intermediates
  mla->q_proj = tensor_create_2d(max_seq_len, n_heads * dk);
  mla->kv_latent = tensor_create_2d(max_seq_len, d_kv_comp);
  mla->concat = tensor_create_2d(max_seq_len, n_heads * dv);
  mla->scores = tensor_create_2d(max_seq_len, max_seq_len);

  mla->rope = rope_create(max_seq_len, dk, 10000.0f);

  return mla;
}

void mla_free(MLA *mla) {
  if (mla) {
    linear_free(mla->w_q);
    linear_free(mla->w_kv_down);
    for (int h = 0; h < mla->n_heads; h++) {
      linear_free(mla->w_k_up[h]);
      linear_free(mla->w_v_up[h]);
      tensor_free(mla->k_proj[h]);
      tensor_free(mla->v_proj[h]);
      tensor_free(mla->head_out[h]);
    }
    free(mla->w_k_up);
    free(mla->w_v_up);
    free(mla->k_proj);
    free(mla->v_proj);
    free(mla->head_out);
    linear_free(mla->w_o);
    tensor_free(mla->q_proj);
    tensor_free(mla->kv_latent);
    tensor_free(mla->concat);
    tensor_free(mla->scores);
    rope_free(mla->rope);
    free(mla);
  }
}

void mla_forward(MLA *mla, Tensor *out, Tensor *x, int *mask, int seq_len) {
  int n_heads = mla->n_heads;
  int dk = mla->dk;
  int dv = mla->dv;

  // project queries
  linear_forward(mla->w_q, mla->q_proj, x, seq_len);

  // compress kv
  linear_forward(mla->w_kv_down, mla->kv_latent, x, seq_len);

  // per-head processing
  for (int h = 0; h < n_heads; h++) {
    // up-project k and v from latent
    linear_forward(mla->w_k_up[h], mla->k_proj[h], mla->kv_latent, seq_len);
    linear_forward(mla->w_v_up[h], mla->v_proj[h], mla->kv_latent, seq_len);

    // extract this head's queries
    Tensor *q_head = tensor_create_2d(seq_len, dk);
    for (int t = 0; t < seq_len; t++) {
      for (int i = 0; i < dk; i++) {
        q_head->data[t * dk + i] =
            mla->q_proj->data[t * (n_heads * dk) + h * dk + i];
      }
    }

    // apply rope to q and k
    rope_apply(mla->rope, q_head, seq_len);
    rope_apply(mla->rope, mla->k_proj[h], seq_len);

    // compute attention scores: Q @ K^T
    matmul_transB(mla->scores->data, q_head->data, mla->k_proj[h]->data,
                  seq_len, dk, seq_len);

    // scale and mask
    for (int i = 0; i < seq_len; i++) {
      for (int j = 0; j < seq_len; j++) {
        float *score = tensor_at_2d(mla->scores, i, j);
        *score *= mla->scale;
        if (mask && !mask[i * seq_len + j]) {
          *score = -1e9f;
        }
      }
    }

    // softmax over each row
    softmax_rows(mla->scores->data, seq_len, seq_len);

    // output: scores @ V
    matmul(mla->head_out[h]->data, mla->scores->data, mla->v_proj[h]->data,
           seq_len, seq_len, dv);

    tensor_free(q_head);
  }

  // concatenate heads
  for (int t = 0; t < seq_len; t++) {
    for (int h = 0; h < n_heads; h++) {
      for (int i = 0; i < dv; i++) {
        mla->concat->data[t * (n_heads * dv) + h * dv + i] =
            mla->head_out[h]->data[t * dv + i];
      }
    }
  }

  // output projection
  linear_forward(mla->w_o, out, mla->concat, seq_len);
}

void mla_backward(MLA *mla, Tensor *dx, Tensor *dy, Tensor *x, int seq_len) {
  // simplified backward - accumulates gradients to weight tensors
  // full implementation would require storing more intermediates

  int n_heads = mla->n_heads;
  int dv = mla->dv;

  // backward through output projection
  Tensor *d_concat = tensor_create_2d(seq_len, n_heads * dv);
  linear_backward(mla->w_o, d_concat, dy, mla->concat, seq_len);

  // backward through kv compression and query projection
  // (simplified: just propagate to input)
  Tensor *d_kv = tensor_create_2d(seq_len, mla->d_kv_comp);
  tensor_zero(d_kv);

  Tensor *d_q = tensor_create_2d(seq_len, n_heads * mla->dk);
  tensor_zero(d_q);

  // accumulate gradients from all heads
  // (this is a simplification - full backward would trace through attention)

  linear_backward(mla->w_q, dx, d_q, x, seq_len);

  Tensor *dx_kv = tensor_create_2d(seq_len, mla->d_model);
  linear_backward(mla->w_kv_down, dx_kv, d_kv, x, seq_len);

  tensor_add_inplace(dx, dx_kv);

  tensor_free(d_concat);
  tensor_free(d_kv);
  tensor_free(d_q);
  tensor_free(dx_kv);
}

// === Expert ===

Expert *expert_create(int d_model, int d_ff) {
  Expert *e = (Expert *)safe_malloc(sizeof(Expert));
  e->d_model = d_model;
  e->d_ff = d_ff;
  e->w_gate = linear_create(d_model, d_ff, 0);
  e->w_value = linear_create(d_model, d_ff, 0);
  e->w_down = linear_create(d_ff, d_model, 0);
  return e;
}

void expert_free(Expert *e) {
  if (e) {
    linear_free(e->w_gate);
    linear_free(e->w_value);
    linear_free(e->w_down);
    free(e);
  }
}

void expert_forward(Expert *e, Tensor *out, Tensor *x, int seq_len) {
  Tensor *gate = tensor_create_2d(seq_len, e->d_ff);
  Tensor *value = tensor_create_2d(seq_len, e->d_ff);
  Tensor *hidden = tensor_create_2d(seq_len, e->d_ff);

  linear_forward(e->w_gate, gate, x, seq_len);
  linear_forward(e->w_value, value, x, seq_len);
  swiglu(hidden->data, gate->data, value->data, seq_len * e->d_ff);
  linear_forward(e->w_down, out, hidden, seq_len);

  tensor_free(gate);
  tensor_free(value);
  tensor_free(hidden);
}

// === MoE ===

MoE *moe_create(int n_experts, int top_k, int d_model, int d_ff,
                int max_seq_len, float aux_loss_coef) {
  MoE *moe = (MoE *)safe_malloc(sizeof(MoE));
  moe->n_experts = n_experts;
  moe->top_k = top_k;
  moe->d_model = d_model;
  moe->d_ff = d_ff;
  moe->aux_loss_coef = aux_loss_coef;

  // allocate experts dynamically
  moe->experts = (Expert **)safe_malloc((size_t)n_experts * sizeof(Expert *));
  for (int i = 0; i < n_experts; i++) {
    moe->experts[i] = expert_create(d_model, d_ff);
  }

  moe->router = linear_create(d_model, n_experts, 0);

  moe->router_logits = tensor_create_2d(max_seq_len, n_experts);
  moe->expert_indices =
      (int *)safe_calloc((size_t)(max_seq_len * top_k), sizeof(int));
  moe->expert_weights =
      (float *)safe_calloc((size_t)(max_seq_len * top_k), sizeof(float));
  moe->expert_counts = (float *)safe_calloc((size_t)n_experts, sizeof(float));
  moe->total_tokens = 0;
  moe->aux_loss = 0.0f;

  moe->expert_out = tensor_create_2d(max_seq_len, d_model);

  return moe;
}

void moe_free(MoE *moe) {
  if (moe) {
    for (int i = 0; i < moe->n_experts; i++) {
      expert_free(moe->experts[i]);
    }
    free(moe->experts);
    linear_free(moe->router);
    tensor_free(moe->router_logits);
    free(moe->expert_indices);
    free(moe->expert_weights);
    free(moe->expert_counts);
    tensor_free(moe->expert_out);
    free(moe);
  }
}

void moe_reset_counts(MoE *moe) {
  memset(moe->expert_counts, 0, (size_t)moe->n_experts * sizeof(float));
  moe->total_tokens = 0;
  moe->aux_loss = 0.0f;
}

static float compute_cov(const float *counts, int n, int total) {
  if (total == 0)
    return 0.0f;

  // compute mean
  float mean = (float)total / (float)n;

  // compute std dev
  float var = 0.0f;
  for (int i = 0; i < n; i++) {
    float diff = counts[i] - mean;
    var += diff * diff;
  }
  var /= (float)n;
  float std = sqrtf(var);

  // coefficient of variation
  if (mean < 1e-8f)
    return 0.0f;
  return std / mean;
}

void moe_forward(MoE *moe, Tensor *out, Tensor *x, int seq_len) {
  int top_k = moe->top_k;
  int d_model = moe->d_model;

  // compute router logits
  linear_forward(moe->router, moe->router_logits, x, seq_len);

  // zero output
  tensor_zero(out);

  // temporary for single-token processing
  Tensor *token_in = tensor_create_2d(1, d_model);
  Tensor *token_out = tensor_create_2d(1, d_model);

  for (int t = 0; t < seq_len; t++) {
    float *logits = moe->router_logits->data + t * moe->n_experts;
    int *indices = moe->expert_indices + t * top_k;
    float *weights = moe->expert_weights + t * top_k;

    // select top-k experts with softmax over selected
    topk_indices(indices, weights, logits, moe->n_experts, top_k);

    // copy input token
    memcpy(token_in->data, x->data + t * d_model,
           (size_t)d_model * sizeof(float));

    // accumulate weighted expert outputs
    for (int k = 0; k < top_k; k++) {
      int expert_idx = indices[k];
      float weight = weights[k];

      // track usage
      moe->expert_counts[expert_idx] += 1.0f;

      // run expert
      expert_forward(moe->experts[expert_idx], token_out, token_in, 1);

      // accumulate to output
      for (int i = 0; i < d_model; i++) {
        out->data[t * d_model + i] += weight * token_out->data[i];
      }
    }
  }

  moe->total_tokens += seq_len;

  // compute auxiliary loss: coefficient of variation of expert usage
  moe->aux_loss =
      moe->aux_loss_coef * compute_cov(moe->expert_counts, moe->n_experts,
                                       moe->total_tokens * top_k);

  tensor_free(token_in);
  tensor_free(token_out);
}

void moe_backward(MoE *moe, Tensor *dx, Tensor *dy, Tensor *x, int seq_len) {
  int top_k = moe->top_k;
  int d_model = moe->d_model;

  tensor_zero(dx);

  // simplified backward: propagate through selected experts
  Tensor *token_in = tensor_create_2d(1, d_model);
  Tensor *token_dy = tensor_create_2d(1, d_model);
  Tensor *token_dx = tensor_create_2d(1, d_model);

  for (int t = 0; t < seq_len; t++) {
    int *indices = moe->expert_indices + t * top_k;
    float *weights = moe->expert_weights + t * top_k;

    memcpy(token_in->data, x->data + t * d_model,
           (size_t)d_model * sizeof(float));

    for (int k = 0; k < top_k; k++) {
      int expert_idx = indices[k];
      float weight = weights[k];

      // scale gradient by weight
      for (int i = 0; i < d_model; i++) {
        token_dy->data[i] = weight * dy->data[t * d_model + i];
      }

      // backward through expert (simplified - just propagate)
      // full implementation would call expert_backward
      matmul_transB(token_dx->data, token_dy->data,
                    moe->experts[expert_idx]->w_down->weight->data, 1, d_model,
                    moe->d_ff);

      // accumulate to dx
      for (int i = 0; i < d_model; i++) {
        dx->data[t * d_model + i] += token_dx->data[i];
      }
    }
  }

  tensor_free(token_in);
  tensor_free(token_dy);
  tensor_free(token_dx);
}

float moe_get_aux_loss(MoE *moe) { return moe->aux_loss; }

// === Embedding ===

Embedding *embedding_create(int vocab_size, int d_model) {
  Embedding *emb = (Embedding *)safe_malloc(sizeof(Embedding));
  emb->vocab_size = vocab_size;
  emb->d_model = d_model;
  emb->scale = sqrtf((float)d_model);
  emb->weight = tensor_create_2d(vocab_size, d_model);
  tensor_init_xavier(emb->weight, vocab_size, d_model);
  return emb;
}

void embedding_free(Embedding *emb) {
  if (emb) {
    tensor_free(emb->weight);
    free(emb);
  }
}

void embedding_forward(Embedding *emb, Tensor *out, int *tokens, int seq_len) {
  for (int t = 0; t < seq_len; t++) {
    int token = tokens[t];
    for (int i = 0; i < emb->d_model; i++) {
      out->data[t * emb->d_model + i] =
          emb->weight->data[token * emb->d_model + i] * emb->scale;
    }
  }
}

void embedding_backward(Embedding *emb, Tensor *dy, int *tokens, int seq_len) {
  tensor_alloc_grad(emb->weight);
  for (int t = 0; t < seq_len; t++) {
    int token = tokens[t];
    for (int i = 0; i < emb->d_model; i++) {
      emb->weight->grad[token * emb->d_model + i] +=
          dy->data[t * emb->d_model + i] * emb->scale;
    }
  }
}
