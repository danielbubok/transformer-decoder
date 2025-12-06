#ifndef TRAIN_H
#define TRAIN_H

#include "transformer.h"
#include <stdio.h>

// adamw optimizer state
typedef struct {
  float *m;
  float *v;
  int size;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  int t;
} AdamW;

// lr schedule state
typedef struct {
  float peak_lr;
  float min_lr;
  int warmup_steps;
  int total_steps;
  int current_step;
} LRSchedule;

// training config
typedef struct {
  float peak_lr;
  int warmup_steps;
  int total_steps;
  float weight_decay;
  float grad_clip_norm;
  int gradient_accumulation_steps;
} TrainConfig;

// optimizer
AdamW *adamw_create(int size, float beta1, float beta2, float eps,
                    float weight_decay);
void adamw_free(AdamW *opt);
void adamw_update(AdamW *opt, float *params, float *grads, int size, float lr);
void adamw_save(AdamW *opt, FILE *f);
int adamw_load(AdamW *opt, FILE *f);

// lr schedule
LRSchedule *lr_schedule_create(float peak_lr, int warmup_steps,
                               int total_steps);
void lr_schedule_free(LRSchedule *sched);
float lr_schedule_get(LRSchedule *sched, int step);

// loss functions
float cross_entropy_loss(float *logits, int *targets, int seq_len,
                         int vocab_size);
void cross_entropy_backward(float *d_logits, float *logits, int *targets,
                            int seq_len, int vocab_size);

// gradient operations
float compute_grad_norm(Transformer *t);
void clip_gradients(Transformer *t, float max_norm);
void accumulate_gradients(Transformer *t, float scale);

// parameter flattening
int count_transformer_params(Transformer *t);
float *flatten_transformer_params(Transformer *t, int *total_size);
void unflatten_transformer_params(Transformer *t, float *params);
float *flatten_transformer_grads(Transformer *t, int *total_size);
void unflatten_transformer_grads(Transformer *t, float *grads);

// training step
void train_step(Transformer *t, AdamW *opt, LRSchedule *sched, int *tokens,
                int seq_len, float *loss_out, float *aux_loss_out);

// generation
int generate_greedy(Transformer *t, int *prompt, int prompt_len, int *out,
                    int max_len, int eos);
int generate_sampled(Transformer *t, int *prompt, int prompt_len, int *out,
                     int max_len, int eos, float temperature, float top_p);

// checkpointing
int save_checkpoint(Transformer *t, AdamW *opt, LRSchedule *sched,
                    const char *path);
int load_checkpoint(Transformer *t, AdamW *opt, LRSchedule *sched,
                    const char *path);

#endif
