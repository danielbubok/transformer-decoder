#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

#define N_HEADS 8

typedef struct {
  Tensor *gamma;
  Tensor *beta;
  int dim;
} LayerNorm;

typedef struct {
  Tensor *weight;
  Tensor *bias;
  int in_dim;
  int out_dim;
} Linear;

typedef struct {
  Tensor *encoding;
  int max_len;
  int dim;
} PositionalEncoding;

typedef struct {
  Tensor *scores;
  Tensor *attn;
  float scale;
} ScaledDotProductAttention;

typedef struct {
  int n_heads;
  int d_model;
  int dk;
  int dv;
  Linear *W_Q[N_HEADS];
  Linear *W_K[N_HEADS];
  Linear *W_V[N_HEADS];
  Linear *W_O;
  Tensor *Q_proj[N_HEADS];
  Tensor *K_proj[N_HEADS];
  Tensor *V_proj[N_HEADS];
  Tensor *head_out[N_HEADS];
  Tensor *concat;
  ScaledDotProductAttention *sdpa;
} MultiHeadAttention;

typedef struct {
  Linear *fc1;
  Linear *fc2;
  Tensor *hidden;
  int d_model;
  int d_ff;
} FeedForward;

typedef struct {
  Tensor *weight;
  int vocab_size;
  int d_model;
  float scale;
} Embedding;

LayerNorm *layernorm_create(int dim);
void layernorm_free(LayerNorm *ln);
void layernorm_forward(LayerNorm *ln, Tensor *x);

Linear *linear_create(int in_dim, int out_dim, int use_bias);
void linear_free(Linear *l);
void linear_forward(Linear *l, Tensor *y, Tensor *x);

PositionalEncoding *positional_encoding_create(int max_len, int dim);
void positional_encoding_free(PositionalEncoding *pe);
void positional_encoding_add(PositionalEncoding *pe, Tensor *x);

ScaledDotProductAttention *sdpa_create(int max_seq_len, int dk);
void sdpa_free(ScaledDotProductAttention *attn);
void sdpa_forward(ScaledDotProductAttention *attn, Tensor *output, Tensor *Q,
                  Tensor *K, Tensor *V, int *mask, int seq_len_q, int seq_len_k,
                  int dk, int dv);

MultiHeadAttention *mha_create(int n_heads, int d_model, int dk, int dv,
                               int max_seq_len);
void mha_free(MultiHeadAttention *mha);
void mha_forward(MultiHeadAttention *mha, Tensor *output, Tensor *Q, Tensor *K,
                 Tensor *V, int *mask, int seq_len_q, int seq_len_k);

FeedForward *ff_create(int d_model, int d_ff, int max_seq_len);
void ff_free(FeedForward *ff);
void ff_forward(FeedForward *ff, Tensor *output, Tensor *input, int seq_len);

Embedding *embedding_create(int vocab_size, int d_model);
void embedding_free(Embedding *emb);
void embedding_forward(Embedding *emb, Tensor *output, int *tokens,
                       int seq_len);

#endif
