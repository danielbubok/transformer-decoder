#include "ops.h"
#include "train.h"
#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EOS_TOKEN 1

typedef struct {
  int *tokens;
  int len;
} Sample;

typedef struct {
  Sample *samples;
  int count;
  int capacity;
} Dataset;

Dataset *dataset_create(void) {
  Dataset *d = (Dataset *)safe_malloc(sizeof(Dataset));
  d->capacity = 16;
  d->count = 0;
  d->samples = (Sample *)safe_malloc((size_t)d->capacity * sizeof(Sample));
  return d;
}

void dataset_add(Dataset *d, int *tokens, int len) {
  if (d->count >= d->capacity) {
    d->capacity *= 2;
    d->samples =
        (Sample *)realloc(d->samples, (size_t)d->capacity * sizeof(Sample));
  }
  d->samples[d->count].tokens = (int *)safe_malloc((size_t)len * sizeof(int));
  memcpy(d->samples[d->count].tokens, tokens, (size_t)len * sizeof(int));
  d->samples[d->count].len = len;
  d->count++;
}

void dataset_free(Dataset *d) {
  if (d) {
    for (int i = 0; i < d->count; i++) {
      free(d->samples[i].tokens);
    }
    free(d->samples);
    free(d);
  }
}

void create_sequence_dataset(Dataset *d, int num_samples) {
  for (int i = 0; i < num_samples; i++) {
    int len = 4 + (i % 4);
    int *tokens = (int *)safe_malloc((size_t)len * sizeof(int));
    for (int j = 0; j < len - 1; j++) {
      tokens[j] = 2 + (j % (VOCAB_SIZE - 2));
    }
    tokens[len - 1] = EOS_TOKEN;
    dataset_add(d, tokens, len);
    free(tokens);
  }
}

void train(Transformer *model, Dataset *data, int epochs, int warmup_steps) {
  int param_count;
  float *params = flatten_transformer_params(model, &param_count);
  free(params);

  AdamState *adam = adam_create(param_count);
  int step = 0;

  printf("training: %d params, %d samples, %d epochs\n", param_count,
         data->count, epochs);

  for (int epoch = 0; epoch < epochs; epoch++) {
    float epoch_loss = 0.0f;
    for (int i = 0; i < data->count; i++) {
      Sample *s = &data->samples[i];
      float lr = get_learning_rate(step, model->d_model, warmup_steps);
      float loss;
      train_step(model, adam, s->tokens, s->len, lr, &loss);
      epoch_loss += loss;
      step++;
    }
    epoch_loss /= (float)data->count;
    printf("epoch %d: loss=%.4f\n", epoch + 1, epoch_loss);
  }

  adam_free(adam);
}

void inference_demo(Transformer *model) {
  printf("\ngeneration:\n");
  int out[MAX_SEQ_LEN];

  int prompt[] = {2, 3};
  int prompt_len = 2;

  printf("prompt: ");
  for (int i = 0; i < prompt_len; i++)
    printf("%d ", prompt[i]);
  printf("\n");

  int gen_len = generate(model, prompt, prompt_len, out, 10, EOS_TOKEN);
  printf("output: ");
  for (int i = 0; i < gen_len; i++)
    printf("%d ", out[i]);
  printf("\n");
}

int main(void) {
  printf("decoder-only transformer\n");
  printf("config: n=%d d_model=%d d_ff=%d h=%d vocab=%d\n\n", N_LAYERS, D_MODEL,
         D_FF, N_HEADS, VOCAB_SIZE);

  Transformer *model = transformer_create(VOCAB_SIZE, D_MODEL, D_FF, N_HEADS,
                                          D_K, D_V, N_LAYERS, MAX_SEQ_LEN);

  Dataset *data = dataset_create();
  create_sequence_dataset(data, 8);

  train(model, data, 5, 100);
  inference_demo(model);

  dataset_free(data);
  transformer_free(model);

  return 0;
}
