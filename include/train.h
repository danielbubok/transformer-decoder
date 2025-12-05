#ifndef TRAIN_H
#define TRAIN_H

#include "transformer.h"

typedef struct {
  float *m;
  float *v;
  int size;
  float beta1;
  float beta2;
  float eps;
  int t;
} AdamState;

AdamState *adam_create(int size);
void adam_free(AdamState *adam);
void adam_update(AdamState *adam, float *param, float *grad, int size,
                 float lr);

float cross_entropy_loss(float *logits, int *targets, int seq_len,
                         int vocab_size);
void cross_entropy_backward(float *d_logits, float *logits, int *targets,
                            int seq_len, int vocab_size);

float get_learning_rate(int step, int d_model, int warmup_steps);

int count_transformer_params(Transformer *t);
float *flatten_transformer_params(Transformer *t, int *total_size);
void unflatten_transformer_params(Transformer *t, float *params);

void train_step(Transformer *t, AdamState *adam, int *tokens, int seq_len,
                float lr, float *loss_out);

int generate(Transformer *t, int *prompt, int prompt_len, int *out, int max_len,
             int eos);

#endif
