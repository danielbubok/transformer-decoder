#include "train.h"
#include "ops.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

AdamState *adam_create(int size) {
  AdamState *adam = (AdamState *)safe_malloc(sizeof(AdamState));
  adam->m = (float *)safe_calloc((size_t)size, sizeof(float));
  adam->v = (float *)safe_calloc((size_t)size, sizeof(float));
  adam->size = size;
  adam->beta1 = 0.999f;
  adam->beta2 = 0.999f;
  adam->eps = 1e-9f;
  adam->t = 0;
  return adam;
}

void adam_free(AdamState *adam) {
  if (adam) {
    free(adam->m);
    free(adam->v);
    free(adam);
  }
}

void adam_update(AdamState *adam, float *param, float *grad, int size,
                 float lr) {
  adam->t++;
  float bc1 = 1.0f - powf(adam->beta1, (float)adam->t);
  float bc2 = 1.0f - powf(adam->beta2, (float)adam->t);
  for (int i = 0; i < size; i++) {
    adam->m[i] = adam->beta1 * adam->m[i] + (1.0f - adam->beta1) * grad[i];
    adam->v[i] =
        adam->beta2 * adam->v[i] + (1.0f - adam->beta2) * grad[i] * grad[i];
    float m_hat = adam->m[i] / bc1;
    float v_hat = adam->v[i] / bc2;
    param[i] -= lr * m_hat / (sqrtf(v_hat) + adam->eps);
  }
}

float get_learning_rate(int step, int d_model, int warmup_steps) {
  float s = (float)(step + 1);
  float w = (float)warmup_steps;
  float d = (float)d_model;
  return powf(d, -0.5f) * fminf(powf(s, -0.5f), s * powf(w, -1.5f));
}

float cross_entropy_loss(float *logits, int *targets, int seq_len,
                         int vocab_size) {
  float loss = 0.0f;
  for (int t = 0; t < seq_len; t++) {
    float *row = logits + t * vocab_size;
    float max_val = row[0];
    for (int i = 1; i < vocab_size; i++) {
      if (row[i] > max_val)
        max_val = row[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
      sum += expf(row[i] - max_val);
    }
    float log_sum = max_val + logf(sum);
    loss -= row[targets[t]] - log_sum;
  }
  return loss / (float)seq_len;
}

void cross_entropy_backward(float *d_logits, float *logits, int *targets,
                            int seq_len, int vocab_size) {
  for (int t = 0; t < seq_len; t++) {
    float *row = logits + t * vocab_size;
    float *d_row = d_logits + t * vocab_size;
    float max_val = row[0];
    for (int i = 1; i < vocab_size; i++) {
      if (row[i] > max_val)
        max_val = row[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
      d_row[i] = expf(row[i] - max_val);
      sum += d_row[i];
    }
    for (int i = 0; i < vocab_size; i++) {
      d_row[i] /= sum;
    }
    d_row[targets[t]] -= 1.0f;
    for (int i = 0; i < vocab_size; i++) {
      d_row[i] /= (float)seq_len;
    }
  }
}

static int count_linear_params(Linear *l) {
  int count = l->in_dim * l->out_dim;
  if (l->bias)
    count += l->out_dim;
  return count;
}

static int count_layernorm_params(LayerNorm *ln) { return ln->dim * 2; }

static int count_mha_params(MultiHeadAttention *mha) {
  int count = 0;
  for (int h = 0; h < mha->n_heads; h++) {
    count += count_linear_params(mha->W_Q[h]);
    count += count_linear_params(mha->W_K[h]);
    count += count_linear_params(mha->W_V[h]);
  }
  count += count_linear_params(mha->W_O);
  return count;
}

static int count_ff_params(FeedForward *ff) {
  return count_linear_params(ff->fc1) + count_linear_params(ff->fc2);
}

static int count_decoder_layer_params(DecoderLayer *layer) {
  return count_mha_params(layer->self_attn) + count_ff_params(layer->ff) +
         count_layernorm_params(layer->norm1) +
         count_layernorm_params(layer->norm2);
}

int count_transformer_params(Transformer *t) {
  int count = t->embedding->vocab_size * t->embedding->d_model;
  count += count_linear_params(t->output_proj);
  for (int i = 0; i < t->decoder->n_layers; i++) {
    count += count_decoder_layer_params(t->decoder->layers[i]);
  }
  return count;
}

static void flatten_linear_params(Linear *l, float *params, int *offset) {
  int wsize = l->in_dim * l->out_dim;
  memcpy(params + *offset, l->weight->data, (size_t)wsize * sizeof(float));
  *offset += wsize;
  if (l->bias) {
    memcpy(params + *offset, l->bias->data, (size_t)l->out_dim * sizeof(float));
    *offset += l->out_dim;
  }
}

static void flatten_layernorm_params(LayerNorm *ln, float *params,
                                     int *offset) {
  memcpy(params + *offset, ln->gamma->data, (size_t)ln->dim * sizeof(float));
  *offset += ln->dim;
  memcpy(params + *offset, ln->beta->data, (size_t)ln->dim * sizeof(float));
  *offset += ln->dim;
}

static void flatten_mha_params(MultiHeadAttention *mha, float *params,
                               int *offset) {
  for (int h = 0; h < mha->n_heads; h++) {
    flatten_linear_params(mha->W_Q[h], params, offset);
    flatten_linear_params(mha->W_K[h], params, offset);
    flatten_linear_params(mha->W_V[h], params, offset);
  }
  flatten_linear_params(mha->W_O, params, offset);
}

static void flatten_ff_params(FeedForward *ff, float *params, int *offset) {
  flatten_linear_params(ff->fc1, params, offset);
  flatten_linear_params(ff->fc2, params, offset);
}

float *flatten_transformer_params(Transformer *t, int *total_size) {
  *total_size = count_transformer_params(t);
  float *params = (float *)safe_malloc((size_t)*total_size * sizeof(float));
  int offset = 0;
  int emb_size = t->embedding->vocab_size * t->embedding->d_model;
  memcpy(params + offset, t->embedding->weight->data,
         (size_t)emb_size * sizeof(float));
  offset += emb_size;
  flatten_linear_params(t->output_proj, params, &offset);
  for (int i = 0; i < t->decoder->n_layers; i++) {
    DecoderLayer *layer = t->decoder->layers[i];
    flatten_mha_params(layer->self_attn, params, &offset);
    flatten_ff_params(layer->ff, params, &offset);
    flatten_layernorm_params(layer->norm1, params, &offset);
    flatten_layernorm_params(layer->norm2, params, &offset);
  }
  return params;
}

static void unflatten_linear_params(Linear *l, float *params, int *offset) {
  int wsize = l->in_dim * l->out_dim;
  memcpy(l->weight->data, params + *offset, (size_t)wsize * sizeof(float));
  *offset += wsize;
  if (l->bias) {
    memcpy(l->bias->data, params + *offset, (size_t)l->out_dim * sizeof(float));
    *offset += l->out_dim;
  }
}

static void unflatten_layernorm_params(LayerNorm *ln, float *params,
                                       int *offset) {
  memcpy(ln->gamma->data, params + *offset, (size_t)ln->dim * sizeof(float));
  *offset += ln->dim;
  memcpy(ln->beta->data, params + *offset, (size_t)ln->dim * sizeof(float));
  *offset += ln->dim;
}

static void unflatten_mha_params(MultiHeadAttention *mha, float *params,
                                 int *offset) {
  for (int h = 0; h < mha->n_heads; h++) {
    unflatten_linear_params(mha->W_Q[h], params, offset);
    unflatten_linear_params(mha->W_K[h], params, offset);
    unflatten_linear_params(mha->W_V[h], params, offset);
  }
  unflatten_linear_params(mha->W_O, params, offset);
}

static void unflatten_ff_params(FeedForward *ff, float *params, int *offset) {
  unflatten_linear_params(ff->fc1, params, offset);
  unflatten_linear_params(ff->fc2, params, offset);
}

void unflatten_transformer_params(Transformer *t, float *params) {
  int offset = 0;
  int emb_size = t->embedding->vocab_size * t->embedding->d_model;
  memcpy(t->embedding->weight->data, params + offset,
         (size_t)emb_size * sizeof(float));
  offset += emb_size;
  unflatten_linear_params(t->output_proj, params, &offset);
  for (int i = 0; i < t->decoder->n_layers; i++) {
    DecoderLayer *layer = t->decoder->layers[i];
    unflatten_mha_params(layer->self_attn, params, &offset);
    unflatten_ff_params(layer->ff, params, &offset);
    unflatten_layernorm_params(layer->norm1, params, &offset);
    unflatten_layernorm_params(layer->norm2, params, &offset);
  }
}

static uint64_t train_rng[2] = {0xdeadbeef12345678ULL, 0x9876543210fedcbaULL};

static float randf(void) {
  uint64_t s1 = train_rng[0];
  uint64_t s0 = train_rng[1];
  uint64_t result = s0 + s1;
  train_rng[0] = s0;
  s1 ^= s1 << 23;
  train_rng[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
  return (float)(result >> 11) * (1.0f / 9007199254740992.0f);
}

static float randn(void) {
  float u1 = randf();
  float u2 = randf();
  if (u1 < 1e-10f)
    u1 = 1e-10f;
  return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

void train_step(Transformer *t, AdamState *adam, int *tokens, int seq_len,
                float lr, float *loss_out) {
  int param_count;
  float *params = flatten_transformer_params(t, &param_count);
  float *grads = (float *)safe_calloc((size_t)param_count, sizeof(float));
  float *noise = (float *)safe_malloc((size_t)param_count * sizeof(float));

  transformer_forward(t, tokens, seq_len);
  float base_loss = cross_entropy_loss(t->logits->data, tokens + 1, seq_len - 1,
                                       t->vocab_size);
  *loss_out = base_loss;

  int n_samples = 8;
  float sigma = 0.01f;

  for (int s = 0; s < n_samples; s++) {
    for (int i = 0; i < param_count; i++) {
      noise[i] = randn() * sigma;
      params[i] += noise[i];
    }
    unflatten_transformer_params(t, params);
    transformer_forward(t, tokens, seq_len);
    float loss = cross_entropy_loss(t->logits->data, tokens + 1, seq_len - 1,
                                    t->vocab_size);

    float weight = (loss - base_loss) / (sigma * sigma);
    for (int i = 0; i < param_count; i++) {
      grads[i] += weight * noise[i] / (float)n_samples;
      params[i] -= noise[i];
    }
  }

  unflatten_transformer_params(t, params);
  adam_update(adam, params, grads, param_count, lr);
  unflatten_transformer_params(t, params);

  free(params);
  free(grads);
  free(noise);
}

int generate(Transformer *t, int *prompt, int prompt_len, int *out, int max_len,
             int eos) {
  int tokens[MAX_SEQ_LEN];
  for (int i = 0; i < prompt_len; i++) {
    tokens[i] = prompt[i];
    out[i] = prompt[i];
  }
  int len = prompt_len;

  while (len < max_len) {
    transformer_forward(t, tokens, len);
    float *last = tensor_at_2d(t->logits, len - 1, 0);
    softmax(last, t->vocab_size);

    int next = 0;
    float max_p = last[0];
    for (int i = 1; i < t->vocab_size; i++) {
      if (last[i] > max_p) {
        max_p = last[i];
        next = i;
      }
    }

    out[len] = next;
    if (next == eos)
      return len + 1;

    tokens[len] = next;
    len++;
  }

  return len;
}
