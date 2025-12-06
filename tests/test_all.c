#include "../include/layers.h"
#include "../include/ops.h"
#include "../include/tensor.h"
#include "../include/tokenizer.h"
#include "../include/train.h"
#include "../include/transformer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) static void name(void)
#define RUN_TEST(name)                                                         \
  do {                                                                         \
    tests_run++;                                                               \
    printf("  %s... ", #name);                                                 \
    name();                                                                    \
    tests_passed++;                                                            \
    printf("pass\n");                                                          \
  } while (0)

#define ASSERT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL\n    assertion failed: %s\n    at %s:%d\n", #cond,          \
             __FILE__, __LINE__);                                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define ASSERT_NEAR(a, b, eps)                                                 \
  do {                                                                         \
    float _a = (a), _b = (b), _eps = (eps);                                    \
    if (fabsf(_a - _b) > _eps) {                                               \
      printf("FAIL\n    %f != %f (eps=%f)\n    at %s:%d\n", _a, _b, _eps,      \
             __FILE__, __LINE__);                                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

TEST(test_tensor_create) {
  Tensor *t = tensor_create_2d(4, 8);
  ASSERT(t != NULL);
  ASSERT(t->shape[0] == 4);
  ASSERT(t->shape[1] == 8);
  ASSERT(t->size == 32);
  tensor_free(t);
}

TEST(test_tensor_zero) {
  Tensor *t = tensor_create_2d(2, 3);
  t->data[0] = 1.0f;
  t->data[5] = 2.0f;
  tensor_zero(t);
  for (int i = 0; i < t->size; i++) {
    ASSERT_NEAR(t->data[i], 0.0f, 1e-9f);
  }
  tensor_free(t);
}

TEST(test_tensor_copy) {
  Tensor *src = tensor_create_2d(2, 2);
  Tensor *dst = tensor_create_2d(2, 2);
  src->data[0] = 1.0f;
  src->data[1] = 2.0f;
  src->data[2] = 3.0f;
  src->data[3] = 4.0f;
  tensor_copy(dst, src);
  for (int i = 0; i < 4; i++) {
    ASSERT_NEAR(dst->data[i], src->data[i], 1e-9f);
  }
  tensor_free(src);
  tensor_free(dst);
}

TEST(test_softmax) {
  float x[] = {1.0f, 2.0f, 3.0f};
  softmax(x, 3);
  float sum = x[0] + x[1] + x[2];
  ASSERT_NEAR(sum, 1.0f, 1e-5f);
  ASSERT(x[2] > x[1]);
  ASSERT(x[1] > x[0]);
}

TEST(test_mish) {
  float x[] = {-1.0f, 0.0f, 1.0f, 2.0f};
  mish_inplace(x, 4);
  ASSERT(x[0] < 0.0f);
  ASSERT_NEAR(x[1], 0.0f, 1e-5f);
  ASSERT(x[2] > 0.8f);
  ASSERT(x[3] > 1.9f);
}

TEST(test_matmul) {
  float A[] = {1, 2, 3, 4};
  float B[] = {5, 6, 7, 8};
  float C[4];
  matmul(C, A, B, 2, 2, 2);
  ASSERT_NEAR(C[0], 19.0f, 1e-5f);
  ASSERT_NEAR(C[1], 22.0f, 1e-5f);
  ASSERT_NEAR(C[2], 43.0f, 1e-5f);
  ASSERT_NEAR(C[3], 50.0f, 1e-5f);
}

TEST(test_tokenizer_encode) {
  const char *text = "hi";
  int tokens[16];
  int n = tokenize(text, tokens, 16);
  ASSERT(n == 2);
  ASSERT(tokens[0] == 'h');
  ASSERT(tokens[1] == 'i');
}

TEST(test_tokenizer_special) {
  const char *text = "<system>a</system>";
  int tokens[16];
  int n = tokenize(text, tokens, 16);
  ASSERT(n == 3);
  ASSERT(tokens[0] == TOK_SYSTEM_OPEN);
  ASSERT(tokens[1] == 'a');
  ASSERT(tokens[2] == TOK_SYSTEM_CLOSE);
}

TEST(test_tokenizer_roundtrip) {
  const char *text = "<think>test</think>";
  int tokens[32];
  int n = tokenize(text, tokens, 32);
  char decoded[64];
  detokenize(tokens, n, decoded, 64);
  ASSERT(strcmp(text, decoded) == 0);
}

TEST(test_linear) {
  Linear *l = linear_create(4, 2, 1);
  Tensor *x = tensor_create_2d(1, 4);
  Tensor *y = tensor_create_2d(1, 2);
  for (int i = 0; i < 4; i++)
    x->data[i] = 1.0f;
  linear_forward(l, y, x);
  ASSERT(y->data[0] != 0.0f || y->data[1] != 0.0f);
  tensor_free(x);
  tensor_free(y);
  linear_free(l);
}

TEST(test_layernorm) {
  LayerNorm *ln = layernorm_create(4);
  Tensor *x = tensor_create_2d(1, 4);
  x->data[0] = 1.0f;
  x->data[1] = 2.0f;
  x->data[2] = 3.0f;
  x->data[3] = 4.0f;
  layernorm_forward(ln, x);
  float mean = 0.0f;
  for (int i = 0; i < 4; i++)
    mean += x->data[i];
  mean /= 4.0f;
  ASSERT_NEAR(mean, 0.0f, 1e-4f);
  tensor_free(x);
  layernorm_free(ln);
}

TEST(test_embedding) {
  Embedding *emb = embedding_create(10, 4);
  Tensor *out = tensor_create_2d(2, 4);
  int tokens[] = {1, 2};
  embedding_forward(emb, out, tokens, 2);
  ASSERT(out->data[0] != 0.0f || out->data[1] != 0.0f);
  tensor_free(out);
  embedding_free(emb);
}

TEST(test_feedforward) {
  FeedForward *ff = ff_create(4, 8, 2);
  Tensor *x = tensor_create_2d(2, 4);
  Tensor *y = tensor_create_2d(2, 4);
  for (int i = 0; i < 8; i++)
    x->data[i] = 0.5f;
  ff_forward(ff, y, x, 2);
  int nonzero = 0;
  for (int i = 0; i < 8; i++) {
    if (y->data[i] != 0.0f)
      nonzero++;
  }
  ASSERT(nonzero > 0);
  tensor_free(x);
  tensor_free(y);
  ff_free(ff);
}

TEST(test_mha) {
  MultiHeadAttention *mha = mha_create(2, 4, 2, 2, 4);
  Tensor *x = tensor_create_2d(2, 4);
  Tensor *out = tensor_create_2d(2, 4);
  for (int i = 0; i < 8; i++)
    x->data[i] = 0.1f * (float)i;
  mha_forward(mha, out, x, x, x, NULL, 2, 2);
  int nonzero = 0;
  for (int i = 0; i < 8; i++) {
    if (fabsf(out->data[i]) > 1e-9f)
      nonzero++;
  }
  ASSERT(nonzero > 0);
  tensor_free(x);
  tensor_free(out);
  mha_free(mha);
}

TEST(test_mla) {
  MultiHeadLatentAttention *mla = mla_create(2, 8, 4, 4, 4, 8);
  Tensor *x = tensor_create_2d(2, 8);
  Tensor *out = tensor_create_2d(2, 8);
  for (int i = 0; i < 16; i++)
    x->data[i] = 0.1f * (float)i;
  mla_forward(mla, out, x, NULL, 2);
  int nonzero = 0;
  for (int i = 0; i < 16; i++) {
    if (fabsf(out->data[i]) > 1e-9f)
      nonzero++;
  }
  ASSERT(nonzero > 0);
  tensor_free(x);
  tensor_free(out);
  mla_free(mla);
}

TEST(test_moe_routing) {
  MixtureOfExperts *moe = moe_create(4, 2, 4, 8, 2);
  Tensor *x = tensor_create_2d(2, 4);
  Tensor *out = tensor_create_2d(2, 4);
  for (int i = 0; i < 8; i++)
    x->data[i] = 0.1f * (float)i;
  moe_forward(moe, out, x, 2);
  int nonzero = 0;
  for (int i = 0; i < 8; i++) {
    if (fabsf(out->data[i]) > 1e-9f)
      nonzero++;
  }
  ASSERT(nonzero > 0);
  tensor_free(x);
  tensor_free(out);
  moe_free(moe);
}

TEST(test_moe_different_topk) {
  for (int top_k = 1; top_k <= 4; top_k++) {
    MixtureOfExperts *moe = moe_create(4, top_k, 4, 8, 2);
    Tensor *x = tensor_create_2d(1, 4);
    Tensor *out = tensor_create_2d(1, 4);
    for (int i = 0; i < 4; i++)
      x->data[i] = 1.0f;
    moe_forward(moe, out, x, 1);
    int nonzero = 0;
    for (int i = 0; i < 4; i++) {
      if (fabsf(out->data[i]) > 1e-9f)
        nonzero++;
    }
    ASSERT(nonzero > 0);
    tensor_free(x);
    tensor_free(out);
    moe_free(moe);
  }
}

TEST(test_rope) {
  RoPE *rope = rope_create(8, 4);
  Tensor *x = tensor_create_2d(2, 4);
  for (int i = 0; i < 8; i++)
    x->data[i] = 1.0f;
  float before[8];
  memcpy(before, x->data, sizeof(before));
  rope_apply(rope, x, 2);
  int changed = 0;
  for (int i = 0; i < 8; i++) {
    if (fabsf(x->data[i] - before[i]) > 1e-6f)
      changed++;
  }
  ASSERT(changed > 0);
  tensor_free(x);
  rope_free(rope);
}

TEST(test_transformer_create_mha_ff) {
  TransformerConfig cfg = {.n_layers = 1,
                           .d_model = 8,
                           .d_ff = 16,
                           .n_heads = 2,
                           .dk = 4,
                           .dv = 4,
                           .d_kv_comp = 2,
                           .max_seq_len = 8,
                           .vocab_size = 32,
                           .use_mla = 0,
                           .use_moe = 0,
                           .n_experts = 2,
                           .top_k = 1};
  Transformer *t = transformer_create(&cfg);
  ASSERT(t != NULL);
  ASSERT(t->decoder->layers[0]->mha != NULL);
  ASSERT(t->decoder->layers[0]->mla == NULL);
  ASSERT(t->decoder->layers[0]->ff != NULL);
  ASSERT(t->decoder->layers[0]->moe == NULL);
  transformer_free(t);
}

TEST(test_transformer_create_mla_moe) {
  TransformerConfig cfg = {.n_layers = 1,
                           .d_model = 8,
                           .d_ff = 16,
                           .n_heads = 2,
                           .dk = 4,
                           .dv = 4,
                           .d_kv_comp = 2,
                           .max_seq_len = 8,
                           .vocab_size = 32,
                           .use_mla = 1,
                           .use_moe = 1,
                           .n_experts = 2,
                           .top_k = 1};
  Transformer *t = transformer_create(&cfg);
  ASSERT(t != NULL);
  ASSERT(t->decoder->layers[0]->mla != NULL);
  ASSERT(t->decoder->layers[0]->mha == NULL);
  ASSERT(t->decoder->layers[0]->moe != NULL);
  ASSERT(t->decoder->layers[0]->ff == NULL);
  transformer_free(t);
}

TEST(test_transformer_forward) {
  TransformerConfig cfg = {.n_layers = 1,
                           .d_model = 8,
                           .d_ff = 16,
                           .n_heads = 2,
                           .dk = 4,
                           .dv = 4,
                           .d_kv_comp = 2,
                           .max_seq_len = 8,
                           .vocab_size = 32,
                           .use_mla = 0,
                           .use_moe = 0,
                           .n_experts = 2,
                           .top_k = 1};
  Transformer *t = transformer_create(&cfg);
  int tokens[] = {1, 2, 3};
  transformer_forward(t, tokens, 3);
  float sum = 0.0f;
  for (int i = 0; i < 3 * 32; i++) {
    sum += fabsf(t->logits->data[i]);
  }
  ASSERT(sum > 0.0f);
  transformer_free(t);
}

TEST(test_cross_entropy) {
  float logits[] = {1.0f, 2.0f, 3.0f, 0.5f, 1.5f, 2.5f};
  int targets[] = {2, 1};
  float loss = cross_entropy_loss(logits, targets, 2, 3);
  ASSERT(loss > 0.0f);
  ASSERT(loss < 10.0f);
}

TEST(test_adam) {
  AdamState *adam = adam_create(4);
  float params[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float grads[] = {0.1f, 0.2f, 0.3f, 0.4f};
  float before[4];
  memcpy(before, params, sizeof(before));
  adam_update(adam, params, grads, 4, 0.01f);
  int changed = 0;
  for (int i = 0; i < 4; i++) {
    if (fabsf(params[i] - before[i]) > 1e-9f)
      changed++;
  }
  ASSERT(changed == 4);
  adam_free(adam);
}

TEST(test_generate) {
  TransformerConfig cfg = {.n_layers = 1,
                           .d_model = 8,
                           .d_ff = 16,
                           .n_heads = 2,
                           .dk = 4,
                           .dv = 4,
                           .d_kv_comp = 2,
                           .max_seq_len = 16,
                           .vocab_size = 32,
                           .use_mla = 0,
                           .use_moe = 0,
                           .n_experts = 2,
                           .top_k = 1};
  Transformer *t = transformer_create(&cfg);
  int prompt[] = {1, 2};
  int out[16];
  int len = generate(t, prompt, 2, out, 8, 0);
  ASSERT(len >= 2);
  ASSERT(len <= 8);
  ASSERT(out[0] == 1);
  ASSERT(out[1] == 2);
  transformer_free(t);
}

TEST(test_generate_sampled) {
  TransformerConfig cfg = {.n_layers = 1,
                           .d_model = 8,
                           .d_ff = 16,
                           .n_heads = 2,
                           .dk = 4,
                           .dv = 4,
                           .d_kv_comp = 2,
                           .max_seq_len = 16,
                           .vocab_size = 32,
                           .use_mla = 0,
                           .use_moe = 0,
                           .n_experts = 2,
                           .top_k = 1};
  Transformer *t = transformer_create(&cfg);
  int prompt[] = {1, 2};
  int out[16];
  int len = generate_sampled(t, prompt, 2, out, 8, 0, 0.8f, 0.9f);
  ASSERT(len >= 2);
  ASSERT(len <= 8);
  ASSERT(out[0] == 1);
  ASSERT(out[1] == 2);
  transformer_free(t);
}

TEST(test_save_load) {
  TransformerConfig cfg = {.n_layers = 1,
                           .d_model = 8,
                           .d_ff = 16,
                           .n_heads = 2,
                           .dk = 4,
                           .dv = 4,
                           .d_kv_comp = 2,
                           .max_seq_len = 8,
                           .vocab_size = 32,
                           .use_mla = 0,
                           .use_moe = 0,
                           .n_experts = 2,
                           .top_k = 1};
  Transformer *t1 = transformer_create(&cfg);
  int pc1;
  float *p1 = flatten_transformer_params(t1, &pc1);
  ASSERT(save_model(t1, "/tmp/test_model.bin") == 0);

  Transformer *t2 = transformer_create(&cfg);
  ASSERT(load_model(t2, "/tmp/test_model.bin") == 0);
  int pc2;
  float *p2 = flatten_transformer_params(t2, &pc2);
  ASSERT(pc1 == pc2);
  for (int i = 0; i < pc1; i++) {
    ASSERT_NEAR(p1[i], p2[i], 1e-9f);
  }

  free(p1);
  free(p2);
  transformer_free(t1);
  transformer_free(t2);
}

TEST(test_param_count) {
  TransformerConfig cfg = {.n_layers = 1,
                           .d_model = 8,
                           .d_ff = 16,
                           .n_heads = 2,
                           .dk = 4,
                           .dv = 4,
                           .d_kv_comp = 2,
                           .max_seq_len = 8,
                           .vocab_size = 32,
                           .use_mla = 0,
                           .use_moe = 0,
                           .n_experts = 2,
                           .top_k = 1};
  Transformer *t = transformer_create(&cfg);
  int count = count_transformer_params(t);
  ASSERT(count > 0);
  int pc;
  float *p = flatten_transformer_params(t, &pc);
  ASSERT(pc == count);
  free(p);
  transformer_free(t);
}

int main(void) {
  printf("running tests...\n\n");

  printf("tensor tests:\n");
  RUN_TEST(test_tensor_create);
  RUN_TEST(test_tensor_zero);
  RUN_TEST(test_tensor_copy);

  printf("\nops tests:\n");
  RUN_TEST(test_softmax);
  RUN_TEST(test_mish);
  RUN_TEST(test_matmul);

  printf("\ntokenizer tests:\n");
  RUN_TEST(test_tokenizer_encode);
  RUN_TEST(test_tokenizer_special);
  RUN_TEST(test_tokenizer_roundtrip);

  printf("\nlayer tests:\n");
  RUN_TEST(test_linear);
  RUN_TEST(test_layernorm);
  RUN_TEST(test_embedding);
  RUN_TEST(test_feedforward);
  RUN_TEST(test_rope);

  printf("\nattention tests:\n");
  RUN_TEST(test_mha);
  RUN_TEST(test_mla);

  printf("\nmoe tests:\n");
  RUN_TEST(test_moe_routing);
  RUN_TEST(test_moe_different_topk);

  printf("\ntransformer tests:\n");
  RUN_TEST(test_transformer_create_mha_ff);
  RUN_TEST(test_transformer_create_mla_moe);
  RUN_TEST(test_transformer_forward);

  printf("\ntraining tests:\n");
  RUN_TEST(test_cross_entropy);
  RUN_TEST(test_adam);
  RUN_TEST(test_param_count);

  printf("\ninference tests:\n");
  RUN_TEST(test_generate);
  RUN_TEST(test_generate_sampled);
  RUN_TEST(test_save_load);

  printf("\n----------------------------------------\n");
  printf("results: %d/%d tests passed\n", tests_passed, tests_run);

  return (tests_passed == tests_run) ? 0 : 1;
}
