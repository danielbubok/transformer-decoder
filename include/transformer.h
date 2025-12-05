#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "layers.h"

#define N_LAYERS 4
#define D_MODEL 64
#define D_FF 256
#define D_K 64
#define D_V 64
#define MAX_SEQ_LEN 128
#define VOCAB_SIZE 8

typedef struct {
  MultiHeadAttention *self_attn;
  FeedForward *ff;
  LayerNorm *norm1;
  LayerNorm *norm2;
  Tensor *attn_out;
  Tensor *ff_out;
  Tensor *residual;
  int d_model;
} DecoderLayer;

typedef struct {
  DecoderLayer *layers[N_LAYERS];
  Tensor *intermediate;
  int n_layers;
  int d_model;
} Decoder;

typedef struct {
  Embedding *embedding;
  PositionalEncoding *pos_encoding;
  Decoder *decoder;
  Linear *output_proj;
  Tensor *embedded;
  Tensor *decoder_out;
  Tensor *logits;
  int *causal_mask;
  int vocab_size;
  int d_model;
  int max_seq_len;
} Transformer;

DecoderLayer *decoder_layer_create(int d_model, int d_ff, int n_heads, int dk,
                                   int dv, int max_seq_len);
void decoder_layer_free(DecoderLayer *layer);
void decoder_layer_forward(DecoderLayer *layer, Tensor *output, Tensor *input,
                           int *causal_mask, int seq_len);

Decoder *decoder_create(int n_layers, int d_model, int d_ff, int n_heads,
                        int dk, int dv, int max_seq_len);
void decoder_free(Decoder *dec);
void decoder_forward(Decoder *dec, Tensor *output, Tensor *input,
                     int *causal_mask, int seq_len);

Transformer *transformer_create(int vocab_size, int d_model, int d_ff,
                                int n_heads, int dk, int dv, int n_layers,
                                int max_seq_len);
void transformer_free(Transformer *t);
void transformer_forward(Transformer *t, int *tokens, int seq_len);

#endif
