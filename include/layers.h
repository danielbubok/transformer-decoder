#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

// no static limits - all arrays dynamically allocated

// rmsnorm replaces layernorm
typedef struct {
  Tensor *gamma;
  int dim;
  float eps;
} RMSNorm;

typedef struct {
  Tensor *weight;
  Tensor *bias;
  int in_dim;
  int out_dim;
  int use_bias;
} Linear;

typedef struct {
  // gate projection: d_model -> d_ff
  Linear *w_gate;
  // value projection: d_model -> d_ff
  Linear *w_value;
  // down projection: d_ff -> d_model
  Linear *w_down;
  int d_model;
  int d_ff;
  // intermediate tensors
  Tensor *gate_out;
  Tensor *value_out;
  Tensor *swiglu_out;
} SwiGLUFFN;

typedef struct {
  Tensor *sin_cache;
  Tensor *cos_cache;
  int max_len;
  int dim;
  float base;
} RoPE;

// multi-head latent attention
typedef struct {
  int n_heads;
  int d_model;
  int dk;        // per-head key/query dim
  int dv;        // per-head value dim
  int d_kv_comp; // latent compression dim

  // query projection: d_model -> n_heads * dk
  Linear *w_q;
  // kv compression: d_model -> d_kv_comp
  Linear *w_kv_down;
  // per-head up-projections from latent (dynamically allocated)
  Linear **w_k_up;
  Linear **w_v_up;
  // output projection: n_heads * dv -> d_model
  Linear *w_o;

  // intermediate tensors (dynamically allocated per head)
  Tensor *q_proj;    // [seq, n_heads * dk]
  Tensor *kv_latent; // [seq, d_kv_comp]
  Tensor **k_proj;   // [seq, dk] per head
  Tensor **v_proj;   // [seq, dv] per head
  Tensor **head_out; // [seq, dv] per head
  Tensor *concat;    // [seq, n_heads * dv]
  Tensor *scores;    // [seq, seq]

  RoPE *rope;
  float scale;
} MLA;

// single expert (swiglu ffn)
typedef struct {
  Linear *w_gate;
  Linear *w_value;
  Linear *w_down;
  int d_model;
  int d_ff;
} Expert;

// mixture of experts layer
typedef struct {
  int n_experts;
  int top_k;
  int d_model;
  int d_ff;
  int max_seq_len;

  Expert **experts; // dynamically allocated array
  Linear *router;   // d_model -> n_experts

  // routing state (dynamically sized for top_k)
  Tensor *router_logits; // [seq, n_experts]
  int *expert_indices;   // [seq, top_k]
  float *expert_weights; // [seq, top_k]

  // expert usage tracking for load balancing
  float *expert_counts; // [n_experts]
  int total_tokens;
  float aux_loss;
  float aux_loss_coef;

  // intermediate
  Tensor *expert_out;
} MoE;

typedef struct {
  Tensor *weight;
  int vocab_size;
  int d_model;
  float scale;
} Embedding;

// rmsnorm
RMSNorm *rmsnorm_create(int dim);
void rmsnorm_free(RMSNorm *rn);
void rmsnorm_forward(RMSNorm *rn, Tensor *out, Tensor *x, int seq_len);
void rmsnorm_backward(RMSNorm *rn, Tensor *dx, Tensor *dy, Tensor *x,
                      int seq_len);

// linear
Linear *linear_create(int in_dim, int out_dim, int use_bias);
void linear_free(Linear *l);
void linear_forward(Linear *l, Tensor *y, Tensor *x, int seq_len);
void linear_backward(Linear *l, Tensor *dx, Tensor *dy, Tensor *x, int seq_len);

// swiglu ffn
SwiGLUFFN *swiglu_ffn_create(int d_model, int d_ff, int max_seq_len);
void swiglu_ffn_free(SwiGLUFFN *ff);
void swiglu_ffn_forward(SwiGLUFFN *ff, Tensor *out, Tensor *x, int seq_len);
void swiglu_ffn_backward(SwiGLUFFN *ff, Tensor *dx, Tensor *dy, Tensor *x,
                         int seq_len);

// rope
RoPE *rope_create(int max_len, int dim, float base);
void rope_free(RoPE *rope);
void rope_apply(RoPE *rope, Tensor *x, int seq_len);

// mla
MLA *mla_create(int n_heads, int d_model, int dk, int dv, int d_kv_comp,
                int max_seq_len);
void mla_free(MLA *mla);
void mla_forward(MLA *mla, Tensor *out, Tensor *x, int *mask, int seq_len);
void mla_backward(MLA *mla, Tensor *dx, Tensor *dy, Tensor *x, int seq_len);

// expert
Expert *expert_create(int d_model, int d_ff);
void expert_free(Expert *e);
void expert_forward(Expert *e, Tensor *out, Tensor *x, int seq_len);

// moe
MoE *moe_create(int n_experts, int top_k, int d_model, int d_ff,
                int max_seq_len, float aux_loss_coef);
void moe_free(MoE *moe);
void moe_forward(MoE *moe, Tensor *out, Tensor *x, int seq_len);
void moe_backward(MoE *moe, Tensor *dx, Tensor *dy, Tensor *x, int seq_len);
float moe_get_aux_loss(MoE *moe);
void moe_reset_counts(MoE *moe);

// embedding
Embedding *embedding_create(int vocab_size, int d_model);
void embedding_free(Embedding *emb);
void embedding_forward(Embedding *emb, Tensor *out, int *tokens, int seq_len);
void embedding_backward(Embedding *emb, Tensor *dy, int *tokens, int seq_len);

#endif
