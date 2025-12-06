#include "train.h"
#include "ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// === AdamW ===

AdamW *adamw_create(int size, float beta1, float beta2, float eps,
                    float weight_decay) {
  AdamW *opt = (AdamW *)safe_malloc(sizeof(AdamW));
  opt->m = (float *)safe_calloc((size_t)size, sizeof(float));
  opt->v = (float *)safe_calloc((size_t)size, sizeof(float));
  opt->size = size;
  opt->beta1 = beta1;
  opt->beta2 = beta2;
  opt->eps = eps;
  opt->weight_decay = weight_decay;
  opt->t = 0;
  return opt;
}

void adamw_free(AdamW *opt) {
  if (opt) {
    free(opt->m);
    free(opt->v);
    free(opt);
  }
}

void adamw_update(AdamW *opt, float *params, float *grads, int size, float lr) {
  opt->t++;
  float bc1 = 1.0f - powf(opt->beta1, (float)opt->t);
  float bc2 = 1.0f - powf(opt->beta2, (float)opt->t);

  for (int i = 0; i < size; i++) {
    float g = grads[i];

    // momentum and variance updates
    opt->m[i] = opt->beta1 * opt->m[i] + (1.0f - opt->beta1) * g;
    opt->v[i] = opt->beta2 * opt->v[i] + (1.0f - opt->beta2) * g * g;

    // bias correction
    float m_hat = opt->m[i] / bc1;
    float v_hat = opt->v[i] / bc2;

    // update with weight decay
    params[i] -= lr * (m_hat / (sqrtf(v_hat) + opt->eps) +
                       opt->weight_decay * params[i]);
  }
}

void adamw_save(AdamW *opt, FILE *f) {
  fwrite(&opt->size, sizeof(int), 1, f);
  fwrite(&opt->beta1, sizeof(float), 1, f);
  fwrite(&opt->beta2, sizeof(float), 1, f);
  fwrite(&opt->eps, sizeof(float), 1, f);
  fwrite(&opt->weight_decay, sizeof(float), 1, f);
  fwrite(&opt->t, sizeof(int), 1, f);
  fwrite(opt->m, sizeof(float), (size_t)opt->size, f);
  fwrite(opt->v, sizeof(float), (size_t)opt->size, f);
}

int adamw_load(AdamW *opt, FILE *f) {
  int size;
  if (fread(&size, sizeof(int), 1, f) != 1)
    return -1;
  if (size != opt->size)
    return -2;

  if (fread(&opt->beta1, sizeof(float), 1, f) != 1)
    return -1;
  if (fread(&opt->beta2, sizeof(float), 1, f) != 1)
    return -1;
  if (fread(&opt->eps, sizeof(float), 1, f) != 1)
    return -1;
  if (fread(&opt->weight_decay, sizeof(float), 1, f) != 1)
    return -1;
  if (fread(&opt->t, sizeof(int), 1, f) != 1)
    return -1;
  if (fread(opt->m, sizeof(float), (size_t)opt->size, f) != (size_t)opt->size)
    return -1;
  if (fread(opt->v, sizeof(float), (size_t)opt->size, f) != (size_t)opt->size)
    return -1;

  return 0;
}

// === LR Schedule ===

LRSchedule *lr_schedule_create(float peak_lr, int warmup_steps,
                               int total_steps) {
  LRSchedule *sched = (LRSchedule *)safe_malloc(sizeof(LRSchedule));
  sched->peak_lr = peak_lr;
  sched->min_lr = peak_lr * 0.1f; // decay to 10% of peak
  sched->warmup_steps = warmup_steps;
  sched->total_steps = total_steps;
  sched->current_step = 0;
  return sched;
}

void lr_schedule_free(LRSchedule *sched) { free(sched); }

float lr_schedule_get(LRSchedule *sched, int step) {
  if (step < sched->warmup_steps) {
    // linear warmup
    return sched->peak_lr * (float)(step + 1) / (float)sched->warmup_steps;
  } else {
    // cosine decay
    int decay_steps = sched->total_steps - sched->warmup_steps;
    int decay_step = step - sched->warmup_steps;
    float progress = (float)decay_step / (float)decay_steps;
    if (progress > 1.0f)
      progress = 1.0f;

    // cosine annealing: lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi *
    // progress))
    return sched->min_lr + 0.5f * (sched->peak_lr - sched->min_lr) *
                               (1.0f + cosf((float)M_PI * progress));
  }
}

// === Loss Functions ===

float cross_entropy_loss(float *logits, int *targets, int seq_len,
                         int vocab_size) {
  float loss = 0.0f;

  for (int t = 0; t < seq_len; t++) {
    float *row = logits + t * vocab_size;
    int target = targets[t];

    // find max for numerical stability
    float max_val = row[0];
    for (int i = 1; i < vocab_size; i++) {
      if (row[i] > max_val)
        max_val = row[i];
    }

    // log-sum-exp
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
      sum += expf(row[i] - max_val);
    }
    float log_sum = max_val + logf(sum);

    loss -= row[target] - log_sum;
  }

  return loss / (float)seq_len;
}

void cross_entropy_backward(float *d_logits, float *logits, int *targets,
                            int seq_len, int vocab_size) {
  for (int t = 0; t < seq_len; t++) {
    float *row = logits + t * vocab_size;
    float *d_row = d_logits + t * vocab_size;
    int target = targets[t];

    // find max
    float max_val = row[0];
    for (int i = 1; i < vocab_size; i++) {
      if (row[i] > max_val)
        max_val = row[i];
    }

    // softmax
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
      d_row[i] = expf(row[i] - max_val);
      sum += d_row[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++) {
      d_row[i] *= inv_sum;
    }
    d_row[target] -= 1.0f;

    // scale by 1/seq_len
    float scale = 1.0f / (float)seq_len;
    for (int i = 0; i < vocab_size; i++) {
      d_row[i] *= scale;
    }
  }
}

// === Gradient Operations ===

static void collect_param_grads(Tensor *t, float **grads, int *offset) {
  if (!t || !t->grad)
    return;
  memcpy(*grads + *offset, t->grad, (size_t)t->size * sizeof(float));
  *offset += t->size;
}

static void collect_linear_grads(Linear *l, float **grads, int *offset) {
  collect_param_grads(l->weight, grads, offset);
  if (l->bias) {
    collect_param_grads(l->bias, grads, offset);
  }
}

float compute_grad_norm(Transformer *t) {
  float norm_sq = 0.0f;

  // embedding
  if (t->embedding->weight->grad) {
    for (int i = 0; i < t->embedding->weight->size; i++) {
      norm_sq += t->embedding->weight->grad[i] * t->embedding->weight->grad[i];
    }
  }

  // output projection
  if (t->output_proj->weight->grad) {
    for (int i = 0; i < t->output_proj->weight->size; i++) {
      norm_sq +=
          t->output_proj->weight->grad[i] * t->output_proj->weight->grad[i];
    }
  }

  // layers
  for (int l = 0; l < t->decoder->n_layers; l++) {
    DecoderLayer *layer = t->decoder->layers[l];

    // norms
    if (layer->norm1->gamma->grad) {
      for (int i = 0; i < layer->norm1->gamma->size; i++) {
        norm_sq += layer->norm1->gamma->grad[i] * layer->norm1->gamma->grad[i];
      }
    }
    if (layer->norm2->gamma->grad) {
      for (int i = 0; i < layer->norm2->gamma->size; i++) {
        norm_sq += layer->norm2->gamma->grad[i] * layer->norm2->gamma->grad[i];
      }
    }

    // mla
    MLA *mla = layer->mla;
    if (mla->w_q->weight->grad) {
      for (int i = 0; i < mla->w_q->weight->size; i++) {
        norm_sq += mla->w_q->weight->grad[i] * mla->w_q->weight->grad[i];
      }
    }
    if (mla->w_kv_down->weight->grad) {
      for (int i = 0; i < mla->w_kv_down->weight->size; i++) {
        norm_sq +=
            mla->w_kv_down->weight->grad[i] * mla->w_kv_down->weight->grad[i];
      }
    }
    if (mla->w_o->weight->grad) {
      for (int i = 0; i < mla->w_o->weight->size; i++) {
        norm_sq += mla->w_o->weight->grad[i] * mla->w_o->weight->grad[i];
      }
    }

    // moe
    MoE *moe = layer->moe;
    if (moe->router->weight->grad) {
      for (int i = 0; i < moe->router->weight->size; i++) {
        norm_sq += moe->router->weight->grad[i] * moe->router->weight->grad[i];
      }
    }
    for (int e = 0; e < moe->n_experts; e++) {
      Expert *exp = moe->experts[e];
      if (exp->w_gate->weight->grad) {
        for (int i = 0; i < exp->w_gate->weight->size; i++) {
          norm_sq +=
              exp->w_gate->weight->grad[i] * exp->w_gate->weight->grad[i];
        }
      }
      if (exp->w_value->weight->grad) {
        for (int i = 0; i < exp->w_value->weight->size; i++) {
          norm_sq +=
              exp->w_value->weight->grad[i] * exp->w_value->weight->grad[i];
        }
      }
      if (exp->w_down->weight->grad) {
        for (int i = 0; i < exp->w_down->weight->size; i++) {
          norm_sq +=
              exp->w_down->weight->grad[i] * exp->w_down->weight->grad[i];
        }
      }
    }
  }

  return sqrtf(norm_sq);
}

static void scale_grads(Tensor *t, float scale) {
  if (!t || !t->grad)
    return;
  for (int i = 0; i < t->size; i++) {
    t->grad[i] *= scale;
  }
}

void clip_gradients(Transformer *t, float max_norm) {
  float norm = compute_grad_norm(t);
  if (norm <= max_norm)
    return;

  float scale = max_norm / norm;

  scale_grads(t->embedding->weight, scale);
  scale_grads(t->output_proj->weight, scale);
  scale_grads(t->final_norm->gamma, scale);

  for (int l = 0; l < t->decoder->n_layers; l++) {
    DecoderLayer *layer = t->decoder->layers[l];
    scale_grads(layer->norm1->gamma, scale);
    scale_grads(layer->norm2->gamma, scale);

    MLA *mla = layer->mla;
    scale_grads(mla->w_q->weight, scale);
    scale_grads(mla->w_kv_down->weight, scale);
    scale_grads(mla->w_o->weight, scale);
    for (int h = 0; h < mla->n_heads; h++) {
      scale_grads(mla->w_k_up[h]->weight, scale);
      scale_grads(mla->w_v_up[h]->weight, scale);
    }

    MoE *moe = layer->moe;
    scale_grads(moe->router->weight, scale);
    for (int e = 0; e < moe->n_experts; e++) {
      scale_grads(moe->experts[e]->w_gate->weight, scale);
      scale_grads(moe->experts[e]->w_value->weight, scale);
      scale_grads(moe->experts[e]->w_down->weight, scale);
    }
  }
}

// === Parameter Flattening ===

static int count_linear_params(Linear *l) {
  int count = l->in_dim * l->out_dim;
  if (l->bias)
    count += l->out_dim;
  return count;
}

static int count_expert_params(Expert *e) {
  return count_linear_params(e->w_gate) + count_linear_params(e->w_value) +
         count_linear_params(e->w_down);
}

static int count_mla_params(MLA *mla) {
  int count = count_linear_params(mla->w_q);
  count += count_linear_params(mla->w_kv_down);
  count += count_linear_params(mla->w_o);
  for (int h = 0; h < mla->n_heads; h++) {
    count += count_linear_params(mla->w_k_up[h]);
    count += count_linear_params(mla->w_v_up[h]);
  }
  return count;
}

static int count_moe_params(MoE *moe) {
  int count = count_linear_params(moe->router);
  for (int e = 0; e < moe->n_experts; e++) {
    count += count_expert_params(moe->experts[e]);
  }
  return count;
}

static int count_layer_params(DecoderLayer *layer) {
  int count = layer->norm1->dim + layer->norm2->dim; // gamma only for rmsnorm
  count += count_mla_params(layer->mla);
  count += count_moe_params(layer->moe);
  return count;
}

int count_transformer_params(Transformer *t) {
  int count = t->embedding->vocab_size * t->embedding->d_model;
  count += count_linear_params(t->output_proj);
  count += t->final_norm->dim;
  for (int i = 0; i < t->decoder->n_layers; i++) {
    count += count_layer_params(t->decoder->layers[i]);
  }
  return count;
}

static void flatten_linear(Linear *l, float *params, int *offset) {
  int wsize = l->in_dim * l->out_dim;
  memcpy(params + *offset, l->weight->data, (size_t)wsize * sizeof(float));
  *offset += wsize;
  if (l->bias) {
    memcpy(params + *offset, l->bias->data, (size_t)l->out_dim * sizeof(float));
    *offset += l->out_dim;
  }
}

static void unflatten_linear(Linear *l, float *params, int *offset) {
  int wsize = l->in_dim * l->out_dim;
  memcpy(l->weight->data, params + *offset, (size_t)wsize * sizeof(float));
  *offset += wsize;
  if (l->bias) {
    memcpy(l->bias->data, params + *offset, (size_t)l->out_dim * sizeof(float));
    *offset += l->out_dim;
  }
}

float *flatten_transformer_params(Transformer *t, int *total_size) {
  *total_size = count_transformer_params(t);
  float *params = (float *)safe_malloc((size_t)*total_size * sizeof(float));
  int offset = 0;

  // embedding
  int emb_size = t->embedding->vocab_size * t->embedding->d_model;
  memcpy(params + offset, t->embedding->weight->data,
         (size_t)emb_size * sizeof(float));
  offset += emb_size;

  // output projection
  flatten_linear(t->output_proj, params, &offset);

  // final norm
  memcpy(params + offset, t->final_norm->gamma->data,
         (size_t)t->final_norm->dim * sizeof(float));
  offset += t->final_norm->dim;

  // layers
  for (int l = 0; l < t->decoder->n_layers; l++) {
    DecoderLayer *layer = t->decoder->layers[l];

    // norms
    memcpy(params + offset, layer->norm1->gamma->data,
           (size_t)layer->norm1->dim * sizeof(float));
    offset += layer->norm1->dim;
    memcpy(params + offset, layer->norm2->gamma->data,
           (size_t)layer->norm2->dim * sizeof(float));
    offset += layer->norm2->dim;

    // mla
    flatten_linear(layer->mla->w_q, params, &offset);
    flatten_linear(layer->mla->w_kv_down, params, &offset);
    flatten_linear(layer->mla->w_o, params, &offset);
    for (int h = 0; h < layer->mla->n_heads; h++) {
      flatten_linear(layer->mla->w_k_up[h], params, &offset);
      flatten_linear(layer->mla->w_v_up[h], params, &offset);
    }

    // moe
    flatten_linear(layer->moe->router, params, &offset);
    for (int e = 0; e < layer->moe->n_experts; e++) {
      flatten_linear(layer->moe->experts[e]->w_gate, params, &offset);
      flatten_linear(layer->moe->experts[e]->w_value, params, &offset);
      flatten_linear(layer->moe->experts[e]->w_down, params, &offset);
    }
  }

  return params;
}

void unflatten_transformer_params(Transformer *t, float *params) {
  int offset = 0;

  // embedding
  int emb_size = t->embedding->vocab_size * t->embedding->d_model;
  memcpy(t->embedding->weight->data, params + offset,
         (size_t)emb_size * sizeof(float));
  offset += emb_size;

  // output projection
  unflatten_linear(t->output_proj, params, &offset);

  // final norm
  memcpy(t->final_norm->gamma->data, params + offset,
         (size_t)t->final_norm->dim * sizeof(float));
  offset += t->final_norm->dim;

  // layers
  for (int l = 0; l < t->decoder->n_layers; l++) {
    DecoderLayer *layer = t->decoder->layers[l];

    // norms
    memcpy(layer->norm1->gamma->data, params + offset,
           (size_t)layer->norm1->dim * sizeof(float));
    offset += layer->norm1->dim;
    memcpy(layer->norm2->gamma->data, params + offset,
           (size_t)layer->norm2->dim * sizeof(float));
    offset += layer->norm2->dim;

    // mla
    unflatten_linear(layer->mla->w_q, params, &offset);
    unflatten_linear(layer->mla->w_kv_down, params, &offset);
    unflatten_linear(layer->mla->w_o, params, &offset);
    for (int h = 0; h < layer->mla->n_heads; h++) {
      unflatten_linear(layer->mla->w_k_up[h], params, &offset);
      unflatten_linear(layer->mla->w_v_up[h], params, &offset);
    }

    // moe
    unflatten_linear(layer->moe->router, params, &offset);
    for (int e = 0; e < layer->moe->n_experts; e++) {
      unflatten_linear(layer->moe->experts[e]->w_gate, params, &offset);
      unflatten_linear(layer->moe->experts[e]->w_value, params, &offset);
      unflatten_linear(layer->moe->experts[e]->w_down, params, &offset);
    }
  }
}

float *flatten_transformer_grads(Transformer *t, int *total_size) {
  *total_size = count_transformer_params(t);
  float *grads = (float *)safe_malloc((size_t)*total_size * sizeof(float));
  int offset = 0;
  float *ptr = grads;

  // embedding
  collect_param_grads(t->embedding->weight, &ptr, &offset);

  // output projection
  collect_linear_grads(t->output_proj, &ptr, &offset);

  // final norm
  collect_param_grads(t->final_norm->gamma, &ptr, &offset);

  // layers
  for (int l = 0; l < t->decoder->n_layers; l++) {
    DecoderLayer *layer = t->decoder->layers[l];

    // norms
    collect_param_grads(layer->norm1->gamma, &ptr, &offset);
    collect_param_grads(layer->norm2->gamma, &ptr, &offset);

    // mla
    collect_linear_grads(layer->mla->w_q, &ptr, &offset);
    collect_linear_grads(layer->mla->w_kv_down, &ptr, &offset);
    collect_linear_grads(layer->mla->w_o, &ptr, &offset);
    for (int h = 0; h < layer->mla->n_heads; h++) {
      collect_linear_grads(layer->mla->w_k_up[h], &ptr, &offset);
      collect_linear_grads(layer->mla->w_v_up[h], &ptr, &offset);
    }

    // moe
    collect_linear_grads(layer->moe->router, &ptr, &offset);
    for (int e = 0; e < layer->moe->n_experts; e++) {
      collect_linear_grads(layer->moe->experts[e]->w_gate, &ptr, &offset);
      collect_linear_grads(layer->moe->experts[e]->w_value, &ptr, &offset);
      collect_linear_grads(layer->moe->experts[e]->w_down, &ptr, &offset);
    }
  }

  return grads;
}

static void distribute_param_grads(Tensor *t, float *grads, int *offset) {
  if (!t || !t->grad)
    return;
  memcpy(t->grad, grads + *offset, (size_t)t->size * sizeof(float));
  *offset += t->size;
}

static void distribute_linear_grads(Linear *l, float *grads, int *offset) {
  distribute_param_grads(l->weight, grads, offset);
  if (l->bias) {
    distribute_param_grads(l->bias, grads, offset);
  }
}

void unflatten_transformer_grads(Transformer *t, float *grads) {
  int offset = 0;

  // embedding
  distribute_param_grads(t->embedding->weight, grads, &offset);

  // output projection
  distribute_linear_grads(t->output_proj, grads, &offset);

  // final norm
  distribute_param_grads(t->final_norm->gamma, grads, &offset);

  // layers
  for (int l = 0; l < t->decoder->n_layers; l++) {
    DecoderLayer *layer = t->decoder->layers[l];

    // norms
    distribute_param_grads(layer->norm1->gamma, grads, &offset);
    distribute_param_grads(layer->norm2->gamma, grads, &offset);

    // mla
    distribute_linear_grads(layer->mla->w_q, grads, &offset);
    distribute_linear_grads(layer->mla->w_kv_down, grads, &offset);
    distribute_linear_grads(layer->mla->w_o, grads, &offset);
    for (int h = 0; h < layer->mla->n_heads; h++) {
      distribute_linear_grads(layer->mla->w_k_up[h], grads, &offset);
      distribute_linear_grads(layer->mla->w_v_up[h], grads, &offset);
    }

    // moe
    distribute_linear_grads(layer->moe->router, grads, &offset);
    for (int e = 0; e < layer->moe->n_experts; e++) {
      distribute_linear_grads(layer->moe->experts[e]->w_gate, grads, &offset);
      distribute_linear_grads(layer->moe->experts[e]->w_value, grads, &offset);
      distribute_linear_grads(layer->moe->experts[e]->w_down, grads, &offset);
    }
  }
}

// === Training Step ===

void train_step(Transformer *t, AdamW *opt, LRSchedule *sched, int *tokens,
                int seq_len, float *loss_out, float *aux_loss_out) {
  t->is_training = 1;

  // zero gradients
  transformer_zero_grad(t);

  // forward pass
  transformer_forward(t, tokens, seq_len);

  // compute loss (predict next token)
  *loss_out = cross_entropy_loss(t->logits->data, tokens + 1, seq_len - 1,
                                 t->vocab_size);
  *aux_loss_out = transformer_get_aux_loss(t);

  // backward pass
  transformer_backward(t, tokens, tokens + 1, seq_len);

  // clip gradients
  clip_gradients(t, 1.0f);

  // get learning rate
  float lr = lr_schedule_get(sched, sched->current_step);
  sched->current_step++;

  // flatten params and grads, update
  int param_count;
  float *params = flatten_transformer_params(t, &param_count);

  // collect gradients
  int grad_count;
  float *grads = flatten_transformer_grads(t, &grad_count);

  if (param_count != grad_count) {
    fprintf(stderr, "error: param count %d != grad count %d\n", param_count,
            grad_count);
    exit(1);
  }

  adamw_update(opt, params, grads, param_count, lr);
  unflatten_transformer_params(t, params);

  free(params);
  free(grads);

  t->is_training = 0;
}

// === Generation ===

int generate_greedy(Transformer *t, int *prompt, int prompt_len, int *out,
                    int max_len, int eos) {
  int *tokens = (int *)safe_malloc((size_t)t->max_seq_len * sizeof(int));

  for (int i = 0; i < prompt_len; i++) {
    tokens[i] = prompt[i];
    out[i] = prompt[i];
  }

  int len = prompt_len;
  t->is_training = 0;

  while (len < max_len && len < t->max_seq_len) {
    transformer_forward(t, tokens, len);

    // get logits for last position
    float *last_logits = t->logits->data + (len - 1) * t->vocab_size;

    // argmax
    int next = 0;
    float max_val = last_logits[0];
    for (int i = 1; i < t->vocab_size; i++) {
      if (last_logits[i] > max_val) {
        max_val = last_logits[i];
        next = i;
      }
    }

    out[len] = next;
    if (next == eos) {
      free(tokens);
      return len + 1;
    }

    tokens[len] = next;
    len++;
  }

  free(tokens);
  return len;
}

int generate_sampled(Transformer *t, int *prompt, int prompt_len, int *out,
                     int max_len, int eos, float temperature, float top_p) {
  int *tokens = (int *)safe_malloc((size_t)t->max_seq_len * sizeof(int));
  float *probs = (float *)safe_malloc((size_t)t->vocab_size * sizeof(float));
  int *indices = (int *)safe_malloc((size_t)t->vocab_size * sizeof(int));

  for (int i = 0; i < prompt_len; i++) {
    tokens[i] = prompt[i];
    out[i] = prompt[i];
  }

  int len = prompt_len;
  t->is_training = 0;

  while (len < max_len && len < t->max_seq_len) {
    transformer_forward(t, tokens, len);

    float *last_logits = t->logits->data + (len - 1) * t->vocab_size;

    // apply temperature
    if (temperature > 0.0f) {
      for (int i = 0; i < t->vocab_size; i++) {
        probs[i] = last_logits[i] / temperature;
      }
    } else {
      memcpy(probs, last_logits, (size_t)t->vocab_size * sizeof(float));
    }

    // softmax
    float max_val = probs[0];
    for (int i = 1; i < t->vocab_size; i++) {
      if (probs[i] > max_val)
        max_val = probs[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < t->vocab_size; i++) {
      probs[i] = expf(probs[i] - max_val);
      sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < t->vocab_size; i++) {
      probs[i] *= inv_sum;
      indices[i] = i;
    }

    // sort by probability (descending)
    for (int i = 0; i < t->vocab_size - 1; i++) {
      for (int j = i + 1; j < t->vocab_size; j++) {
        if (probs[j] > probs[i]) {
          float tmp = probs[i];
          probs[i] = probs[j];
          probs[j] = tmp;
          int ti = indices[i];
          indices[i] = indices[j];
          indices[j] = ti;
        }
      }
    }

    // nucleus sampling
    float cumsum = 0.0f;
    int nucleus_size = 0;
    for (int i = 0; i < t->vocab_size; i++) {
      cumsum += probs[i];
      nucleus_size++;
      if (cumsum >= top_p)
        break;
    }

    // renormalize and sample
    float renorm = 0.0f;
    for (int i = 0; i < nucleus_size; i++) {
      renorm += probs[i];
    }

    float r = rng_uniform() * renorm;
    cumsum = 0.0f;
    int next = indices[0];
    for (int i = 0; i < nucleus_size; i++) {
      cumsum += probs[i];
      if (r < cumsum) {
        next = indices[i];
        break;
      }
    }

    out[len] = next;
    if (next == eos) {
      free(tokens);
      free(probs);
      free(indices);
      return len + 1;
    }

    tokens[len] = next;
    len++;
  }

  free(tokens);
  free(probs);
  free(indices);
  return len;
}

// === Checkpointing ===

#define CHECKPOINT_MAGIC 0x4D4F4543 // "MOEC"
#define CHECKPOINT_VERSION 1

int save_checkpoint(Transformer *t, AdamW *opt, LRSchedule *sched,
                    const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;

  // magic and version
  int magic = CHECKPOINT_MAGIC;
  int version = CHECKPOINT_VERSION;
  fwrite(&magic, sizeof(int), 1, f);
  fwrite(&version, sizeof(int), 1, f);

  // config
  fwrite(&t->config, sizeof(TransformerConfig), 1, f);

  // parameters
  int param_count;
  float *params = flatten_transformer_params(t, &param_count);
  fwrite(&param_count, sizeof(int), 1, f);
  fwrite(params, sizeof(float), (size_t)param_count, f);
  free(params);

  // optimizer state
  adamw_save(opt, f);

  // lr schedule
  fwrite(&sched->peak_lr, sizeof(float), 1, f);
  fwrite(&sched->min_lr, sizeof(float), 1, f);
  fwrite(&sched->warmup_steps, sizeof(int), 1, f);
  fwrite(&sched->total_steps, sizeof(int), 1, f);
  fwrite(&sched->current_step, sizeof(int), 1, f);

  // rng state for determinism
  extern uint64_t rng_state[4];
  fwrite(rng_state, sizeof(uint64_t), 4, f);

  fclose(f);
  return 0;
}

int load_checkpoint(Transformer *t, AdamW *opt, LRSchedule *sched,
                    const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return -1;

  // magic and version
  int magic, version;
  if (fread(&magic, sizeof(int), 1, f) != 1 || magic != CHECKPOINT_MAGIC) {
    fclose(f);
    return -2;
  }
  if (fread(&version, sizeof(int), 1, f) != 1 ||
      version != CHECKPOINT_VERSION) {
    fclose(f);
    return -3;
  }

  // config (verify matches)
  TransformerConfig cfg;
  if (fread(&cfg, sizeof(TransformerConfig), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (cfg.vocab_size != t->config.vocab_size ||
      cfg.d_model != t->config.d_model || cfg.n_layers != t->config.n_layers ||
      cfg.n_experts != t->config.n_experts) {
    fclose(f);
    return -4;
  }

  // parameters
  int param_count;
  if (fread(&param_count, sizeof(int), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  float *params = (float *)safe_malloc((size_t)param_count * sizeof(float));
  if (fread(params, sizeof(float), (size_t)param_count, f) !=
      (size_t)param_count) {
    free(params);
    fclose(f);
    return -1;
  }
  unflatten_transformer_params(t, params);
  free(params);

  // optimizer state
  if (adamw_load(opt, f) != 0) {
    fclose(f);
    return -5;
  }

  // lr schedule
  if (fread(&sched->peak_lr, sizeof(float), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (fread(&sched->min_lr, sizeof(float), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (fread(&sched->warmup_steps, sizeof(int), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (fread(&sched->total_steps, sizeof(int), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (fread(&sched->current_step, sizeof(int), 1, f) != 1) {
    fclose(f);
    return -1;
  }

  // rng state
  extern uint64_t rng_state[4];
  if (fread(rng_state, sizeof(uint64_t), 4, f) != 4) {
    fclose(f);
    return -1;
  }

  fclose(f);
  return 0;
}
