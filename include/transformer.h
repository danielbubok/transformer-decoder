#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "layers.h"

#define MAX_LAYERS 16

typedef struct {
  int n_layers;
  int d_model;
  int d_ff;
  int n_heads;
  int dk;
  int dv;
  int d_kv_comp;
  int max_seq_len;
  int vocab_size;
  int n_experts;
  int top_k;
  float aux_loss_coef;
} TransformerConfig;

typedef struct {
  MLA *mla;
  MoE *moe;
  RMSNorm *norm1;
  RMSNorm *norm2;
  Tensor *attn_out;
  Tensor *moe_out;
  Tensor *residual1;
  Tensor *residual2;
  Tensor *norm1_out;
  Tensor *norm2_out;
  int d_model;
} DecoderLayer;

typedef struct {
  DecoderLayer *layers[MAX_LAYERS];
  Tensor *intermediate;
  int n_layers;
  int d_model;
} Decoder;

typedef struct {
  Embedding *embedding;
  Decoder *decoder;
  Linear *output_proj;
  RMSNorm *final_norm;
  Tensor *embedded;
  Tensor *decoder_out;
  Tensor *norm_out;
  Tensor *logits;
  int *causal_mask;
  int vocab_size;
  int d_model;
  int max_seq_len;
  TransformerConfig config;

  // training state
  float total_aux_loss;
  int is_training;

  // gradient tensors
  Tensor *d_logits;
  Tensor *d_decoder_out;
  Tensor *d_embedded;
} Transformer;

// decoder layer
DecoderLayer *decoder_layer_create(TransformerConfig *cfg);
void decoder_layer_free(DecoderLayer *layer);
void decoder_layer_forward(DecoderLayer *layer, Tensor *out, Tensor *x,
                           int *mask, int seq_len);
void decoder_layer_backward(DecoderLayer *layer, Tensor *dx, Tensor *dy,
                            Tensor *x, int seq_len);

// decoder
Decoder *decoder_create(TransformerConfig *cfg);
void decoder_free(Decoder *dec);
void decoder_forward(Decoder *dec, Tensor *out, Tensor *x, int *mask,
                     int seq_len);
void decoder_backward(Decoder *dec, Tensor *dx, Tensor *dy, Tensor *x,
                      int seq_len);

// transformer
Transformer *transformer_create(TransformerConfig *cfg);
void transformer_free(Transformer *t);
void transformer_forward(Transformer *t, int *tokens, int seq_len);
void transformer_backward(Transformer *t, int *tokens, int *targets,
                          int seq_len);
void transformer_zero_grad(Transformer *t);
float transformer_get_aux_loss(Transformer *t);

// causal mask
void build_causal_mask(int *mask, int seq_len, int max_len);

#endif
