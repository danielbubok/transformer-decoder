#include "transformer.h"
#include "ops.h"
#include <stdlib.h>
#include <string.h>

DecoderLayer *decoder_layer_create(int d_model, int d_ff, int n_heads, int dk,
                                   int dv, int max_seq_len) {
  DecoderLayer *layer = (DecoderLayer *)safe_malloc(sizeof(DecoderLayer));
  layer->d_model = d_model;
  layer->self_attn = mha_create(n_heads, d_model, dk, dv, max_seq_len);
  layer->ff = ff_create(d_model, d_ff, max_seq_len);
  layer->norm1 = layernorm_create(d_model);
  layer->norm2 = layernorm_create(d_model);
  layer->attn_out = tensor_create_2d(max_seq_len, d_model);
  layer->ff_out = tensor_create_2d(max_seq_len, d_model);
  layer->residual = tensor_create_2d(max_seq_len, d_model);
  return layer;
}

void decoder_layer_free(DecoderLayer *layer) {
  if (layer) {
    mha_free(layer->self_attn);
    ff_free(layer->ff);
    layernorm_free(layer->norm1);
    layernorm_free(layer->norm2);
    tensor_free(layer->attn_out);
    tensor_free(layer->ff_out);
    tensor_free(layer->residual);
    free(layer);
  }
}

void decoder_layer_forward(DecoderLayer *layer, Tensor *output, Tensor *input,
                           int *causal_mask, int seq_len) {
  mha_forward(layer->self_attn, layer->attn_out, input, input, input,
              causal_mask, seq_len, seq_len);
  for (int i = 0; i < seq_len * layer->d_model; i++) {
    layer->residual->data[i] = input->data[i] + layer->attn_out->data[i];
  }
  tensor_copy(layer->attn_out, layer->residual);
  layernorm_forward(layer->norm1, layer->attn_out);
  ff_forward(layer->ff, layer->ff_out, layer->attn_out, seq_len);
  for (int i = 0; i < seq_len * layer->d_model; i++) {
    output->data[i] = layer->attn_out->data[i] + layer->ff_out->data[i];
  }
  layernorm_forward(layer->norm2, output);
}

Decoder *decoder_create(int n_layers, int d_model, int d_ff, int n_heads,
                        int dk, int dv, int max_seq_len) {
  Decoder *dec = (Decoder *)safe_malloc(sizeof(Decoder));
  dec->n_layers = n_layers;
  dec->d_model = d_model;
  dec->intermediate = tensor_create_2d(max_seq_len, d_model);
  for (int i = 0; i < n_layers; i++) {
    dec->layers[i] =
        decoder_layer_create(d_model, d_ff, n_heads, dk, dv, max_seq_len);
  }
  return dec;
}

void decoder_free(Decoder *dec) {
  if (dec) {
    for (int i = 0; i < dec->n_layers; i++) {
      decoder_layer_free(dec->layers[i]);
    }
    tensor_free(dec->intermediate);
    free(dec);
  }
}

void decoder_forward(Decoder *dec, Tensor *output, Tensor *input,
                     int *causal_mask, int seq_len) {
  tensor_copy(dec->intermediate, input);
  for (int i = 0; i < dec->n_layers; i++) {
    if (i == dec->n_layers - 1) {
      decoder_layer_forward(dec->layers[i], output, dec->intermediate,
                            causal_mask, seq_len);
    } else {
      decoder_layer_forward(dec->layers[i], dec->intermediate,
                            dec->intermediate, causal_mask, seq_len);
    }
  }
}

static void update_causal_mask(int *mask, int seq_len, int max_len) {
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < max_len; j++) {
      mask[i * max_len + j] = (j <= i) ? 1 : 0;
    }
  }
}

Transformer *transformer_create(int vocab_size, int d_model, int d_ff,
                                int n_heads, int dk, int dv, int n_layers,
                                int max_seq_len) {
  Transformer *t = (Transformer *)safe_malloc(sizeof(Transformer));
  t->vocab_size = vocab_size;
  t->d_model = d_model;
  t->max_seq_len = max_seq_len;
  t->embedding = embedding_create(vocab_size, d_model);
  t->pos_encoding = positional_encoding_create(max_seq_len, d_model);
  t->decoder =
      decoder_create(n_layers, d_model, d_ff, n_heads, dk, dv, max_seq_len);
  t->output_proj = linear_create(d_model, vocab_size, 0);
  for (int i = 0; i < vocab_size; i++) {
    for (int j = 0; j < d_model; j++) {
      *tensor_at_2d(t->output_proj->weight, j, i) =
          *tensor_at_2d(t->embedding->weight, i, j);
    }
  }
  t->embedded = tensor_create_2d(max_seq_len, d_model);
  t->decoder_out = tensor_create_2d(max_seq_len, d_model);
  t->logits = tensor_create_2d(max_seq_len, vocab_size);
  t->causal_mask =
      (int *)safe_calloc((size_t)(max_seq_len * max_seq_len), sizeof(int));
  return t;
}

void transformer_free(Transformer *t) {
  if (t) {
    embedding_free(t->embedding);
    positional_encoding_free(t->pos_encoding);
    decoder_free(t->decoder);
    linear_free(t->output_proj);
    tensor_free(t->embedded);
    tensor_free(t->decoder_out);
    tensor_free(t->logits);
    free(t->causal_mask);
    free(t);
  }
}

void transformer_forward(Transformer *t, int *tokens, int seq_len) {
  embedding_forward(t->embedding, t->embedded, tokens, seq_len);
  positional_encoding_add(t->pos_encoding, t->embedded);
  update_causal_mask(t->causal_mask, seq_len, t->max_seq_len);
  decoder_forward(t->decoder, t->decoder_out, t->embedded, t->causal_mask,
                  seq_len);
  linear_forward(t->output_proj, t->logits, t->decoder_out);
}
