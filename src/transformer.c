#include "transformer.h"
#include "ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void build_causal_mask(int *mask, int seq_len, int max_len) {
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < max_len; j++) {
      mask[i * max_len + j] = (j <= i) ? 1 : 0;
    }
  }
}

// === DecoderLayer ===

DecoderLayer *decoder_layer_create(TransformerConfig *cfg) {
  DecoderLayer *layer = (DecoderLayer *)safe_malloc(sizeof(DecoderLayer));
  layer->d_model = cfg->d_model;

  layer->mla = mla_create(cfg->n_heads, cfg->d_model, cfg->dk, cfg->dv,
                          cfg->d_kv_comp, cfg->max_seq_len);
  layer->moe = moe_create(cfg->n_experts, cfg->top_k, cfg->d_model, cfg->d_ff,
                          cfg->max_seq_len, cfg->aux_loss_coef);

  layer->norm1 = rmsnorm_create(cfg->d_model);
  layer->norm2 = rmsnorm_create(cfg->d_model);

  layer->attn_out = tensor_create_2d(cfg->max_seq_len, cfg->d_model);
  layer->moe_out = tensor_create_2d(cfg->max_seq_len, cfg->d_model);
  layer->residual1 = tensor_create_2d(cfg->max_seq_len, cfg->d_model);
  layer->residual2 = tensor_create_2d(cfg->max_seq_len, cfg->d_model);
  layer->norm1_out = tensor_create_2d(cfg->max_seq_len, cfg->d_model);
  layer->norm2_out = tensor_create_2d(cfg->max_seq_len, cfg->d_model);

  return layer;
}

void decoder_layer_free(DecoderLayer *layer) {
  if (layer) {
    mla_free(layer->mla);
    moe_free(layer->moe);
    rmsnorm_free(layer->norm1);
    rmsnorm_free(layer->norm2);
    tensor_free(layer->attn_out);
    tensor_free(layer->moe_out);
    tensor_free(layer->residual1);
    tensor_free(layer->residual2);
    tensor_free(layer->norm1_out);
    tensor_free(layer->norm2_out);
    free(layer);
  }
}

void decoder_layer_forward(DecoderLayer *layer, Tensor *out, Tensor *x,
                           int *mask, int seq_len) {
  int size = seq_len * layer->d_model;

  // pre-norm attention
  rmsnorm_forward(layer->norm1, layer->norm1_out, x, seq_len);
  mla_forward(layer->mla, layer->attn_out, layer->norm1_out, mask, seq_len);

  // residual
  for (int i = 0; i < size; i++) {
    layer->residual1->data[i] = x->data[i] + layer->attn_out->data[i];
  }

  // pre-norm moe
  rmsnorm_forward(layer->norm2, layer->norm2_out, layer->residual1, seq_len);
  moe_forward(layer->moe, layer->moe_out, layer->norm2_out, seq_len);

  // residual
  for (int i = 0; i < size; i++) {
    out->data[i] = layer->residual1->data[i] + layer->moe_out->data[i];
  }
}

void decoder_layer_backward(DecoderLayer *layer, Tensor *dx, Tensor *dy,
                            Tensor *x, int seq_len) {
  int size = seq_len * layer->d_model;

  // backward through moe + residual
  Tensor *d_norm2_out = tensor_create_2d(seq_len, layer->d_model);
  moe_backward(layer->moe, d_norm2_out, dy, layer->norm2_out, seq_len);

  // backward through norm2
  Tensor *d_residual1 = tensor_create_2d(seq_len, layer->d_model);
  rmsnorm_backward(layer->norm2, d_residual1, d_norm2_out, layer->residual1,
                   seq_len);

  // add residual grad
  for (int i = 0; i < size; i++) {
    d_residual1->data[i] += dy->data[i];
  }

  // backward through attention + residual
  Tensor *d_norm1_out = tensor_create_2d(seq_len, layer->d_model);
  mla_backward(layer->mla, d_norm1_out, d_residual1, layer->norm1_out, seq_len);

  // backward through norm1
  rmsnorm_backward(layer->norm1, dx, d_norm1_out, x, seq_len);

  // add residual grad
  for (int i = 0; i < size; i++) {
    dx->data[i] += d_residual1->data[i];
  }

  tensor_free(d_norm2_out);
  tensor_free(d_residual1);
  tensor_free(d_norm1_out);
}

// === Decoder ===

Decoder *decoder_create(TransformerConfig *cfg) {
  Decoder *dec = (Decoder *)safe_malloc(sizeof(Decoder));
  dec->n_layers = cfg->n_layers;
  dec->d_model = cfg->d_model;
  dec->intermediate = tensor_create_2d(cfg->max_seq_len, cfg->d_model);

  for (int i = 0; i < cfg->n_layers; i++) {
    dec->layers[i] = decoder_layer_create(cfg);
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

void decoder_forward(Decoder *dec, Tensor *out, Tensor *x, int *mask,
                     int seq_len) {
  tensor_copy(dec->intermediate, x);

  for (int i = 0; i < dec->n_layers; i++) {
    if (i == dec->n_layers - 1) {
      decoder_layer_forward(dec->layers[i], out, dec->intermediate, mask,
                            seq_len);
    } else {
      decoder_layer_forward(dec->layers[i], dec->intermediate,
                            dec->intermediate, mask, seq_len);
    }
  }
}

void decoder_backward(Decoder *dec, Tensor *dx, Tensor *dy, Tensor *x,
                      int seq_len) {
  // backward through layers in reverse order
  Tensor *layer_dx = tensor_create_2d(seq_len, dec->d_model);
  Tensor *layer_dy = tensor_clone(dy);

  for (int i = dec->n_layers - 1; i >= 0; i--) {
    Tensor *layer_x = (i == 0) ? x : dec->intermediate;
    decoder_layer_backward(dec->layers[i], layer_dx, layer_dy, layer_x,
                           seq_len);

    if (i > 0) {
      tensor_copy(layer_dy, layer_dx);
    }
  }

  tensor_copy(dx, layer_dx);

  tensor_free(layer_dx);
  tensor_free(layer_dy);
}

// === Transformer ===

Transformer *transformer_create(TransformerConfig *cfg) {
  Transformer *t = (Transformer *)safe_malloc(sizeof(Transformer));
  t->config = *cfg;
  t->vocab_size = cfg->vocab_size;
  t->d_model = cfg->d_model;
  t->max_seq_len = cfg->max_seq_len;
  t->is_training = 0;
  t->total_aux_loss = 0.0f;

  t->embedding = embedding_create(cfg->vocab_size, cfg->d_model);
  t->decoder = decoder_create(cfg);
  t->output_proj = linear_create(cfg->d_model, cfg->vocab_size, 0);
  t->final_norm = rmsnorm_create(cfg->d_model);

  // tie embedding weights to output projection
  for (int i = 0; i < cfg->vocab_size; i++) {
    for (int j = 0; j < cfg->d_model; j++) {
      t->output_proj->weight->data[j * cfg->vocab_size + i] =
          t->embedding->weight->data[i * cfg->d_model + j];
    }
  }

  t->embedded = tensor_create_2d(cfg->max_seq_len, cfg->d_model);
  t->decoder_out = tensor_create_2d(cfg->max_seq_len, cfg->d_model);
  t->norm_out = tensor_create_2d(cfg->max_seq_len, cfg->d_model);
  t->logits = tensor_create_2d(cfg->max_seq_len, cfg->vocab_size);
  t->causal_mask = (int *)safe_calloc(
      (size_t)(cfg->max_seq_len * cfg->max_seq_len), sizeof(int));

  // gradient tensors
  t->d_logits = tensor_create_2d(cfg->max_seq_len, cfg->vocab_size);
  t->d_decoder_out = tensor_create_2d(cfg->max_seq_len, cfg->d_model);
  t->d_embedded = tensor_create_2d(cfg->max_seq_len, cfg->d_model);

  return t;
}

void transformer_free(Transformer *t) {
  if (t) {
    embedding_free(t->embedding);
    decoder_free(t->decoder);
    linear_free(t->output_proj);
    rmsnorm_free(t->final_norm);
    tensor_free(t->embedded);
    tensor_free(t->decoder_out);
    tensor_free(t->norm_out);
    tensor_free(t->logits);
    free(t->causal_mask);
    tensor_free(t->d_logits);
    tensor_free(t->d_decoder_out);
    tensor_free(t->d_embedded);
    free(t);
  }
}

void transformer_forward(Transformer *t, int *tokens, int seq_len) {
  // reset moe counts for training
  if (t->is_training) {
    for (int i = 0; i < t->decoder->n_layers; i++) {
      moe_reset_counts(t->decoder->layers[i]->moe);
    }
  }

  // embedding
  embedding_forward(t->embedding, t->embedded, tokens, seq_len);

  // build causal mask
  build_causal_mask(t->causal_mask, seq_len, t->max_seq_len);

  // decoder
  decoder_forward(t->decoder, t->decoder_out, t->embedded, t->causal_mask,
                  seq_len);

  // final norm
  rmsnorm_forward(t->final_norm, t->norm_out, t->decoder_out, seq_len);

  // output projection
  linear_forward(t->output_proj, t->logits, t->norm_out, seq_len);

  // accumulate aux loss
  if (t->is_training) {
    t->total_aux_loss = 0.0f;
    for (int i = 0; i < t->decoder->n_layers; i++) {
      t->total_aux_loss += moe_get_aux_loss(t->decoder->layers[i]->moe);
    }
  }
}

void transformer_backward(Transformer *t, int *tokens, int *targets,
                          int seq_len) {
  // compute cross-entropy gradient
  tensor_zero(t->d_logits);

  // softmax(logits) - one_hot(targets)
  for (int pos = 0; pos < seq_len - 1; pos++) {
    float *logits_row = t->logits->data + pos * t->vocab_size;
    float *d_row = t->d_logits->data + pos * t->vocab_size;
    int target = targets[pos];

    // compute softmax
    float max_val = logits_row[0];
    for (int i = 1; i < t->vocab_size; i++) {
      if (logits_row[i] > max_val)
        max_val = logits_row[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < t->vocab_size; i++) {
      d_row[i] = expf(logits_row[i] - max_val);
      sum += d_row[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < t->vocab_size; i++) {
      d_row[i] *= inv_sum;
    }
    d_row[target] -= 1.0f;

    // scale by 1/num_positions
    float scale = 1.0f / (float)(seq_len - 1);
    for (int i = 0; i < t->vocab_size; i++) {
      d_row[i] *= scale;
    }
  }

  // backward through output projection
  Tensor *d_norm_out = tensor_create_2d(seq_len, t->d_model);
  linear_backward(t->output_proj, d_norm_out, t->d_logits, t->norm_out,
                  seq_len);

  // backward through final norm
  rmsnorm_backward(t->final_norm, t->d_decoder_out, d_norm_out, t->decoder_out,
                   seq_len);

  // backward through decoder
  decoder_backward(t->decoder, t->d_embedded, t->d_decoder_out, t->embedded,
                   seq_len);

  // backward through embedding
  embedding_backward(t->embedding, t->d_embedded, tokens, seq_len);

  tensor_free(d_norm_out);
}

void transformer_zero_grad(Transformer *t) {
  tensor_zero_grad(t->embedding->weight);
  tensor_zero_grad(t->output_proj->weight);
  tensor_zero_grad(t->final_norm->gamma);

  for (int l = 0; l < t->decoder->n_layers; l++) {
    DecoderLayer *layer = t->decoder->layers[l];
    tensor_zero_grad(layer->norm1->gamma);
    tensor_zero_grad(layer->norm2->gamma);

    // mla weights
    tensor_zero_grad(layer->mla->w_q->weight);
    tensor_zero_grad(layer->mla->w_kv_down->weight);
    tensor_zero_grad(layer->mla->w_o->weight);
    for (int h = 0; h < layer->mla->n_heads; h++) {
      tensor_zero_grad(layer->mla->w_k_up[h]->weight);
      tensor_zero_grad(layer->mla->w_v_up[h]->weight);
    }

    // moe weights
    tensor_zero_grad(layer->moe->router->weight);
    for (int e = 0; e < layer->moe->n_experts; e++) {
      tensor_zero_grad(layer->moe->experts[e]->w_gate->weight);
      tensor_zero_grad(layer->moe->experts[e]->w_value->weight);
      tensor_zero_grad(layer->moe->experts[e]->w_down->weight);
    }
  }
}

float transformer_get_aux_loss(Transformer *t) { return t->total_aux_loss; }
