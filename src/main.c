#include "memmap.h"
#include "opencl.h"
#include "ops.h"
#include "precision.h"
#include "shard.h"
#include "tensor.h"
#include "tokenizer.h"
#include "train.h"
#include "transformer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ANSI_CLEAR_LINE "\r\033[K"
#define ANSI_GREEN "\033[32m"
#define ANSI_RESET "\033[0m"

// === Utility Functions ===

static void print_usage(const char *prog) {
  printf("usage: %s <command> [options]\n\n", prog);
  printf("commands:\n");
  printf("  train      train the model on text data\n");
  printf("  generate   generate text from a prompt\n");
  printf("  test       run comprehensive test suite\n");
  printf("  info       display model information\n");
  printf("  estimate   estimate memory requirements\n");
  printf("\nmodel architecture (train/generate):\n");
  printf("  --layers <n>        number of decoder layers (default: 2)\n");
  printf("  --d-model <n>       model dimension (default: 32)\n");
  printf("  --d-ff <n>          feedforward dimension (default: 64)\n");
  printf("  --n-heads <n>       attention heads (default: 2)\n");
  printf("  --dk <n>            key dimension per head (default: 4)\n");
  printf("  --dv <n>            value dimension per head (default: 4)\n");
  printf("  --d-kv-comp <n>     kv compression dimension (default: 2)\n");
  printf("  --n-experts <n>     number of experts (default: 128)\n");
  printf("  --top-k <n>         top-k expert selection (default: 8)\n");
  printf("  --aux-loss <f>      aux loss coefficient (default: 0.01)\n");
  printf("  --precision <type>  fp32|fp16|bf16|fp8 (default: fp32)\n");
  printf("\ntrain options:\n");
  printf("  --data <path>       training data file\n");
  printf("  --epochs <n>        number of epochs (default: 1)\n");
  printf("  --lr <float>        peak learning rate (default: 1e-4)\n");
  printf("  --batch <n>         batch size (default: 1)\n");
  printf("  --accum <n>         gradient accumulation steps (default: 1)\n");
  printf("  --checkpoint <path> checkpoint path\n");
  printf("  --seq-len <n>       sequence length (default: 1024)\n");
  printf("  --shards <n>        number of checkpoint shards (default: 1)\n");
  printf("  --shard-dir <path>  sharded checkpoint directory\n");
  printf("\ngenerate options:\n");
  printf("  --model <path>      model checkpoint\n");
  printf("  --prompt <text>     starting prompt\n");
  printf("  --max-tokens <n>    maximum tokens to generate (default: 256)\n");
  printf("  --temperature <f>   sampling temperature (default: 0.8)\n");
  printf("  --top-p <f>         nucleus sampling threshold (default: 0.9)\n");
  printf("\nestimate options:\n");
  printf("  (uses model architecture flags above)\n");
  printf("  --batch <n>         batch size for activation estimation\n");
}

static void progress_bar(int current, int total, float loss, float aux_loss,
                         float lr) {
  int width = 40;
  int filled = (current * width) / total;

  printf(ANSI_CLEAR_LINE "[");
  for (int i = 0; i < width; i++) {
    if (i < filled) {
      printf("=");
    } else if (i == filled) {
      printf(">");
    } else {
      printf(" ");
    }
  }
  printf("] %d/%d loss=%.4f aux=%.4f lr=%.2e", current, total, loss, aux_loss,
         lr);
  fflush(stdout);
}

// === Model Configuration ===

static TransformerConfig get_default_config(void) {
  return (TransformerConfig){.n_layers = 2,
                             .d_model = 32,
                             .d_ff = 64,
                             .n_heads = 2,
                             .dk = 4,
                             .dv = 4,
                             .d_kv_comp = 2,
                             .max_seq_len = 1024,
                             .vocab_size = TOK_VOCAB_SIZE,
                             .n_experts = 128,
                             .top_k = 8,
                             .aux_loss_coef = 0.01f};
}

static void parse_config_from_args(TransformerConfig *cfg, int argc,
                                   char **argv) {
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
      cfg->n_layers = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--d-model") == 0 && i + 1 < argc) {
      cfg->d_model = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--d-ff") == 0 && i + 1 < argc) {
      cfg->d_ff = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--n-heads") == 0 && i + 1 < argc) {
      cfg->n_heads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--dk") == 0 && i + 1 < argc) {
      cfg->dk = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--dv") == 0 && i + 1 < argc) {
      cfg->dv = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--d-kv-comp") == 0 && i + 1 < argc) {
      cfg->d_kv_comp = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--n-experts") == 0 && i + 1 < argc) {
      cfg->n_experts = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
      cfg->top_k = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--aux-loss") == 0 && i + 1 < argc) {
      cfg->aux_loss_coef = (float)atof(argv[++i]);
    } else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) {
      cfg->max_seq_len = atoi(argv[++i]);
    }
  }
}

// === Data Loading ===

typedef struct {
  int *tokens;
  int len;
} Sequence;

typedef struct {
  Sequence *sequences;
  int count;
  int capacity;
} Dataset;

static Dataset *dataset_create(void) {
  Dataset *d = (Dataset *)safe_malloc(sizeof(Dataset));
  d->capacity = 16;
  d->count = 0;
  d->sequences =
      (Sequence *)safe_malloc((size_t)d->capacity * sizeof(Sequence));
  return d;
}

static void dataset_add(Dataset *d, int *tokens, int len) {
  if (d->count >= d->capacity) {
    d->capacity *= 2;
    d->sequences = (Sequence *)safe_realloc(d->sequences, (size_t)d->capacity *
                                                              sizeof(Sequence));
  }
  d->sequences[d->count].tokens = (int *)safe_malloc((size_t)len * sizeof(int));
  memcpy(d->sequences[d->count].tokens, tokens, (size_t)len * sizeof(int));
  d->sequences[d->count].len = len;
  d->count++;
}

static void dataset_free(Dataset *d) {
  if (d) {
    for (int i = 0; i < d->count; i++) {
      free(d->sequences[i].tokens);
    }
    free(d->sequences);
    free(d);
  }
}

static Dataset *load_text_file(const char *path, int seq_len) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "failed to open: %s\n", path);
    return (void *)0;
  }

  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  char *text = (char *)safe_malloc((size_t)file_size + 1);
  if (fread(text, 1, (size_t)file_size, f) != (size_t)file_size) {
    free(text);
    fclose(f);
    return (void *)0;
  }
  text[file_size] = '\0';
  fclose(f);

  // tokenize entire file
  int *all_tokens = (int *)safe_malloc((size_t)file_size * sizeof(int));
  int total_tokens = tokenize(text, all_tokens, (int)file_size);
  free(text);

  if (total_tokens < 0) {
    free(all_tokens);
    return (void *)0;
  }

  // pack into sequences
  Dataset *d = dataset_create();
  int offset = 0;
  while (offset + seq_len <= total_tokens) {
    dataset_add(d, all_tokens + offset, seq_len);
    offset += seq_len;
  }

  free(all_tokens);
  printf("loaded %d sequences of length %d from %s\n", d->count, seq_len, path);
  return d;
}

// === Commands ===

static int cmd_train(int argc, char **argv) {
  const char *data_path = (void *)0;
  const char *checkpoint_path = (void *)0;
  int epochs = 1;
  float peak_lr = 1e-4f;
  int seq_len = 1024;

  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
      data_path = argv[++i];
    } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
      epochs = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
      peak_lr = (float)atof(argv[++i]);
    } else if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
      checkpoint_path = argv[++i];
    } else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) {
      seq_len = atoi(argv[++i]);
    }
  }

  if (!data_path) {
    fprintf(stderr, "error: --data is required\n");
    return 1;
  }

  // initialize opencl
  if (cl_init()) {
    printf("opencl acceleration enabled\n");
  } else {
    printf("using cpu fallback\n");
  }

  // load data
  Dataset *data = load_text_file(data_path, seq_len);
  if (!data || data->count == 0) {
    fprintf(stderr, "error: failed to load training data\n");
    cl_cleanup();
    return 1;
  }

  // create model with cli overrides
  TransformerConfig cfg = get_default_config();
  parse_config_from_args(&cfg, argc, argv);
  cfg.max_seq_len = seq_len;
  Transformer *model = transformer_create(&cfg);

  int param_count = count_transformer_params(model);
  printf("model: %d params, %d layers, d_model=%d, %d experts, %d heads\\n",
         param_count, cfg.n_layers, cfg.d_model, cfg.n_experts, cfg.n_heads);

  // create optimizer and schedule
  int total_steps = epochs * data->count;
  int warmup_steps = (int)(0.05f * (float)total_steps);

  AdamW *opt = adamw_create(param_count, 0.9f, 0.999f, 1e-8f, 0.01f);
  LRSchedule *sched = lr_schedule_create(peak_lr, warmup_steps, total_steps);

  // load checkpoint if provided
  if (checkpoint_path) {
    if (load_checkpoint(model, opt, sched, checkpoint_path) == 0) {
      printf("resumed from checkpoint: %s\n", checkpoint_path);
    }
  }

  // training loop
  printf("training for %d epochs (%d steps, %d warmup)\n", epochs, total_steps,
         warmup_steps);

  int step = sched->current_step;
  for (int epoch = 0; epoch < epochs; epoch++) {
    float epoch_loss = 0.0f;
    float epoch_aux = 0.0f;

    for (int i = 0; i < data->count; i++) {
      float loss, aux_loss;
      float lr = lr_schedule_get(sched, step);

      train_step(model, opt, sched, data->sequences[i].tokens,
                 data->sequences[i].len, &loss, &aux_loss);

      epoch_loss += loss;
      epoch_aux += aux_loss;
      step++;

      progress_bar(i + 1, data->count, loss, aux_loss, lr);
    }

    epoch_loss /= (float)data->count;
    epoch_aux /= (float)data->count;

    printf("\nepoch %d complete: avg_loss=%.4f avg_aux=%.4f\n", epoch + 1,
           epoch_loss, epoch_aux);

    // save checkpoint
    if (checkpoint_path) {
      char epoch_path[256];
      snprintf(epoch_path, sizeof(epoch_path), "%s.epoch%d", checkpoint_path,
               epoch + 1);
      if (save_checkpoint(model, opt, sched, epoch_path) == 0) {
        printf("saved checkpoint: %s\n", epoch_path);
      }
    }
  }

  // final save
  if (checkpoint_path) {
    if (save_checkpoint(model, opt, sched, checkpoint_path) == 0) {
      printf("saved final checkpoint: %s\n", checkpoint_path);
    }
  }

  // cleanup
  transformer_free(model);
  adamw_free(opt);
  lr_schedule_free(sched);
  dataset_free(data);
  cl_cleanup();

  return 0;
}

static int cmd_generate(int argc, char **argv) {
  const char *model_path = (void *)0;
  const char *prompt = (void *)0;
  int max_tokens = 256;
  float temperature = 0.8f;
  float top_p = 0.9f;

  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
      prompt = argv[++i];
    } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
      max_tokens = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      temperature = (float)atof(argv[++i]);
    } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
      top_p = (float)atof(argv[++i]);
    }
  }

  if (!model_path) {
    fprintf(stderr, "error: --model is required\n");
    return 1;
  }

  // create model
  TransformerConfig cfg = get_default_config();
  Transformer *model = transformer_create(&cfg);

  // load checkpoint
  int param_count = count_transformer_params(model);
  AdamW *opt = adamw_create(param_count, 0.9f, 0.999f, 1e-8f, 0.01f);
  LRSchedule *sched = lr_schedule_create(1e-4f, 100, 1000);

  if (load_checkpoint(model, opt, sched, model_path) != 0) {
    fprintf(stderr, "error: failed to load model from %s\n", model_path);
    transformer_free(model);
    adamw_free(opt);
    lr_schedule_free(sched);
    return 1;
  }

  printf("loaded model from %s\n", model_path);

  // tokenize prompt
  int *prompt_tokens =
      (int *)safe_malloc((size_t)cfg.max_seq_len * sizeof(int));
  int prompt_len = 0;

  if (prompt) {
    prompt_len = tokenize(prompt, prompt_tokens, cfg.max_seq_len);
    if (prompt_len < 0) {
      fprintf(stderr, "error: failed to tokenize prompt\n");
      free(prompt_tokens);
      transformer_free(model);
      adamw_free(opt);
      lr_schedule_free(sched);
      return 1;
    }
  }

  // generate
  int *output = (int *)safe_malloc((size_t)cfg.max_seq_len * sizeof(int));
  int gen_len;

  if (temperature > 0.0f) {
    gen_len = generate_sampled(model, prompt_tokens, prompt_len, output,
                               max_tokens, -1, temperature, top_p);
  } else {
    gen_len = generate_greedy(model, prompt_tokens, prompt_len, output,
                              max_tokens, -1);
  }

  // stream output token by token
  char token_str[16];
  for (int i = prompt_len; i < gen_len; i++) {
    int tok = output[i];

    if (is_special_token(tok)) {
      const char *special = get_special_token_str(tok);
      if (special) {
        printf("%s", special);
      }
    } else if (tok >= 0 && tok < 256) {
      token_str[0] = (char)tok;
      token_str[1] = '\0';
      printf("%s", token_str);
    }
    fflush(stdout);
  }
  printf("\n");

  // cleanup
  free(prompt_tokens);
  free(output);
  transformer_free(model);
  adamw_free(opt);
  lr_schedule_free(sched);

  return 0;
}

static int cmd_info(int argc, char **argv) {
  TransformerConfig cfg = get_default_config();
  parse_config_from_args(&cfg, argc, argv);

  printf("moe transformer configuration\n");
  printf("-----------------------------\n");
  printf("layers:        %d\n", cfg.n_layers);
  printf("model dim:     %d\n", cfg.d_model);
  printf("ff dim:        %d\n", cfg.d_ff);
  printf("heads:         %d\n", cfg.n_heads);
  printf("head dim (dk): %d\n", cfg.dk);
  printf("head dim (dv): %d\n", cfg.dv);
  printf("kv latent:     %d\n", cfg.d_kv_comp);
  printf("max seq len:   %d\n", cfg.max_seq_len);
  printf("vocab size:    %d\n", cfg.vocab_size);
  printf("experts:       %d\n", cfg.n_experts);
  printf("top-k:         %d\n", cfg.top_k);
  printf("aux loss coef: %.3f\n", cfg.aux_loss_coef);

  // estimate parameters
  Transformer *model = transformer_create(&cfg);
  int param_count = count_transformer_params(model);
  transformer_free(model);

  printf("\ntotal parameters: %d (%.2f M)\n", param_count,
         (float)param_count / 1e6f);
  printf("memory estimate:  %.2f MB (fp32)\n",
         (float)param_count * 4.0f / 1e6f);

  return 0;
}

static int cmd_estimate(int argc, char **argv) {
  TransformerConfig cfg = get_default_config();
  parse_config_from_args(&cfg, argc, argv);

  int batch_size = 1;
  int precision_bytes = 4; // fp32 default

  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
      batch_size = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--precision") == 0 && i + 1 < argc) {
      i++;
      DType dtype = dtype_from_string(argv[i]);
      precision_bytes = dtype_size(dtype);
    }
  }

  printf("memory estimation for model:\n");
  printf("  layers:     %d\n", cfg.n_layers);
  printf("  d_model:    %d\n", cfg.d_model);
  printf("  d_ff:       %d\n", cfg.d_ff);
  printf("  n_heads:    %d\n", cfg.n_heads);
  printf("  n_experts:  %d\n", cfg.n_experts);
  printf("  top_k:      %d\n", cfg.top_k);
  printf("  max_seq:    %d\n", cfg.max_seq_len);
  printf("  batch_size: %d\n", batch_size);
  printf("  precision:  %d bytes\n\n", precision_bytes);

  MemoryEstimate est = estimate_memory(
      cfg.n_layers, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_experts,
      cfg.top_k, cfg.vocab_size, cfg.max_seq_len, batch_size, precision_bytes);
  print_memory_estimate(&est);

  return 0;
}

// forward declaration
static int run_tests(void);

static int cmd_test(int argc, char **argv) {
  (void)argc;
  (void)argv;
  return run_tests();
}

// === Main ===

int main(int argc, char **argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  const char *cmd = argv[1];

  // seed rng with time for non-deterministic runs
  // note: checkpoint loading will override this for determinism
  rng_seed((uint64_t)time((void *)0));

  if (strcmp(cmd, "train") == 0) {
    return cmd_train(argc - 2, argv + 2);
  } else if (strcmp(cmd, "generate") == 0) {
    return cmd_generate(argc - 2, argv + 2);
  } else if (strcmp(cmd, "test") == 0) {
    return cmd_test(argc - 2, argv + 2);
  } else if (strcmp(cmd, "info") == 0) {
    return cmd_info(argc - 2, argv + 2);
  } else if (strcmp(cmd, "estimate") == 0) {
    return cmd_estimate(argc - 2, argv + 2);
  } else if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
    print_usage(argv[0]);
    return 0;
  } else {
    fprintf(stderr, "unknown command: %s\\n", cmd);
    print_usage(argv[0]);
    return 1;
  }
}

// === Comprehensive Test Suite ===

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) static void name(void)
#define RUN_TEST(name)                                                         \
  do {                                                                         \
    tests_run++;                                                               \
    printf("  %s... ", #name);                                                 \
    fflush(stdout);                                                            \
    name();                                                                    \
    tests_passed++;                                                            \
    printf(ANSI_GREEN "pass" ANSI_RESET "\n");                                 \
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
    float _a = (float)(a), _b = (float)(b), _eps = (float)(eps);               \
    if (fabsf(_a - _b) > _eps) {                                               \
      printf("FAIL\n    %e != %e (diff=%e, eps=%e)\n    at %s:%d\n", _a, _b,   \
             fabsf(_a - _b), _eps, __FILE__, __LINE__);                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// tokenizer tests
TEST(test_tokenizer_byte) {
  int tokens[8];
  int n = tokenize("abc", tokens, 8);
  ASSERT(n == 3);
  ASSERT(tokens[0] == 'a');
  ASSERT(tokens[1] == 'b');
  ASSERT(tokens[2] == 'c');
}

TEST(test_tokenizer_special) {
  int tokens[8];
  int n = tokenize("<system>x</system>", tokens, 8);
  ASSERT(n == 3);
  ASSERT(tokens[0] == TOK_SYSTEM_OPEN);
  ASSERT(tokens[1] == 'x');
  ASSERT(tokens[2] == TOK_SYSTEM_CLOSE);
}

TEST(test_tokenizer_roundtrip) {
  const char *text = "<think>hello world</think>";
  int tokens[64];
  int n = tokenize(text, tokens, 64);
  ASSERT(n > 0);

  char decoded[128];
  int len = detokenize(tokens, n, decoded, 128);
  ASSERT(len > 0);
  ASSERT(strcmp(text, decoded) == 0);
}

TEST(test_tokenizer_empty) {
  int tokens[8];
  int n = tokenize("", tokens, 8);
  ASSERT(n == 0);
}

TEST(test_tokenizer_binary) {
  char binary[] = {(char)0x00, (char)0x01, (char)0xff, (char)0x80, (char)0x00};
  int tokens[8];
  int n = tokenize(binary, tokens, 8);
  // note: strlen will stop at first null byte
  // so this only gets us to the first 0x00
  ASSERT(n == 0); // empty due to null terminator
}

// matmul tests
TEST(test_matmul_basic) {
  float A[] = {1, 2, 3, 4};
  float B[] = {5, 6, 7, 8};
  float C[4];
  matmul(C, A, B, 2, 2, 2);
  ASSERT_NEAR(C[0], 19.0f, 1e-6f);
  ASSERT_NEAR(C[1], 22.0f, 1e-6f);
  ASSERT_NEAR(C[2], 43.0f, 1e-6f);
  ASSERT_NEAR(C[3], 50.0f, 1e-6f);
}

TEST(test_matmul_transB) {
  float A[] = {1, 2, 3, 4};
  float B[] = {5, 7, 6, 8}; // transposed: [[5,6],[7,8]]
  float C[4];
  matmul_transB(C, A, B, 2, 2, 2);
  ASSERT_NEAR(C[0], 19.0f, 1e-6f);
  ASSERT_NEAR(C[1], 22.0f, 1e-6f);
  ASSERT_NEAR(C[2], 43.0f, 1e-6f);
  ASSERT_NEAR(C[3], 50.0f, 1e-6f);
}

// rmsnorm tests
TEST(test_rmsnorm_forward) {
  float x[] = {1, 2, 3, 4};
  float gamma[] = {1, 1, 1, 1};
  float out[4];
  rmsnorm_cpu(out, x, gamma, 1, 4, 1e-6f);

  // rms = sqrt((1+4+9+16)/4) = sqrt(7.5)
  // normalized should have rms = 1
  float ms = 0;
  for (int i = 0; i < 4; i++) {
    ms += out[i] * out[i];
  }
  ms /= 4.0f;
  ASSERT_NEAR(sqrtf(ms), 1.0f, 1e-5f);
}

TEST(test_rmsnorm_backward) {
  float x[] = {1, 2, 3, 4};
  float gamma[] = {1, 1, 1, 1};
  float dy[] = {0.1f, 0.2f, 0.3f, 0.4f};
  float dx[4];
  float dgamma[4];

  rmsnorm_backward_cpu(dx, dgamma, dy, x, gamma, 1, 4, 1e-6f);

  // numerical gradient check
  float eps = 1e-4f;
  for (int i = 0; i < 4; i++) {
    float x_plus[4], x_minus[4];
    float out_plus[4], out_minus[4];
    memcpy(x_plus, x, sizeof(x));
    memcpy(x_minus, x, sizeof(x));
    x_plus[i] += eps;
    x_minus[i] -= eps;

    rmsnorm_cpu(out_plus, x_plus, gamma, 1, 4, 1e-6f);
    rmsnorm_cpu(out_minus, x_minus, gamma, 1, 4, 1e-6f);

    float numerical_grad = 0;
    for (int j = 0; j < 4; j++) {
      numerical_grad += dy[j] * (out_plus[j] - out_minus[j]) / (2 * eps);
    }

    ASSERT_NEAR(dx[i], numerical_grad, 1e-3f);
  }
}

// rope tests
TEST(test_rope_rotation) {
  float x[8];
  for (int i = 0; i < 8; i++)
    x[i] = 1.0f;

  rope_forward(x, 2, 4, 10000.0f);

  // check that values changed (rotation applied)
  int changed = 0;
  for (int i = 0; i < 8; i++) {
    if (fabsf(x[i] - 1.0f) > 1e-6f)
      changed++;
  }
  ASSERT(changed > 0);
}

TEST(test_rope_orthogonal) {
  // rope should preserve vector norms
  float x[] = {1, 2, 3, 4};
  float norm_before = l2_norm(x, 4);

  rope_forward(x, 1, 4, 10000.0f);
  float norm_after = l2_norm(x, 4);

  ASSERT_NEAR(norm_before, norm_after, 1e-5f);
}

// swiglu tests
TEST(test_swiglu_forward) {
  float gate[] = {0.0f, 1.0f, -1.0f, 2.0f};
  float value[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float out[4];

  swiglu(out, gate, value, 4);

  // silu(0) = 0, silu(1) ~= 0.731, silu(-1) ~= -0.269, silu(2) ~= 1.762
  ASSERT_NEAR(out[0], 0.0f, 1e-5f);
  ASSERT(out[1] > 0.7f && out[1] < 0.8f);
  ASSERT(out[2] > -0.3f && out[2] < -0.2f);
  ASSERT(out[3] > 1.7f && out[3] < 1.8f);
}

TEST(test_swiglu_backward) {
  float gate[] = {0.5f, 1.0f};
  float value[] = {1.0f, 2.0f};
  float dout[] = {1.0f, 1.0f};
  float dgate[2], dvalue[2];

  swiglu_backward(dgate, dvalue, dout, gate, value, 2);

  // numerical gradient check
  float eps = 1e-4f;
  for (int i = 0; i < 2; i++) {
    float g_plus[2], g_minus[2];
    float out_plus[2], out_minus[2];
    memcpy(g_plus, gate, sizeof(gate));
    memcpy(g_minus, gate, sizeof(gate));
    g_plus[i] += eps;
    g_minus[i] -= eps;

    swiglu(out_plus, g_plus, value, 2);
    swiglu(out_minus, g_minus, value, 2);

    float numerical = (out_plus[i] - out_minus[i]) / (2 * eps);
    ASSERT_NEAR(dgate[i], numerical * dout[i], 1e-3f);
  }
}

// mla tests
TEST(test_mla_forward) {
  MLA *mla = mla_create(2, 8, 4, 4, 4, 8);
  Tensor *x = tensor_create_2d(2, 8);
  Tensor *out = tensor_create_2d(2, 8);

  for (int i = 0; i < 16; i++)
    x->data[i] = 0.1f * (float)i;

  mla_forward(mla, out, x, (void *)0, 2);

  // check output is non-zero
  float sum = 0;
  for (int i = 0; i < 16; i++)
    sum += fabsf(out->data[i]);
  ASSERT(sum > 0);

  tensor_free(x);
  tensor_free(out);
  mla_free(mla);
}

TEST(test_mla_kv_compression) {
  // verify that kv is actually compressed
  MLA *mla = mla_create(2, 16, 8, 8, 4, 8);

  // d_model=16, d_kv_comp=4, so compression ratio is 4x
  ASSERT(mla->d_kv_comp < mla->d_model);
  ASSERT(mla->kv_latent->shape[1] == 4);

  mla_free(mla);
}

// moe tests
TEST(test_moe_routing) {
  MoE *moe = moe_create(8, 2, 4, 8, 4, 0.01f);
  Tensor *x = tensor_create_2d(2, 4);
  Tensor *out = tensor_create_2d(2, 4);

  for (int i = 0; i < 8; i++)
    x->data[i] = 0.1f * (float)i;

  moe_forward(moe, out, x, 2);

  // check that exactly top_k experts were used per token
  // (check indices are valid)
  for (int t = 0; t < 2; t++) {
    for (int k = 0; k < 2; k++) {
      int idx = moe->expert_indices[t * 2 + k];
      ASSERT(idx >= 0 && idx < 8);
    }
  }

  tensor_free(x);
  tensor_free(out);
  moe_free(moe);
}

TEST(test_moe_load_balance) {
  MoE *moe = moe_create(4, 2, 4, 8, 16, 0.01f);
  Tensor *x = tensor_create_2d(16, 4);
  Tensor *out = tensor_create_2d(16, 4);

  // uniform input
  for (int i = 0; i < 64; i++)
    x->data[i] = 0.1f;

  moe_forward(moe, out, x, 16);

  // aux loss should be computed
  float aux = moe_get_aux_loss(moe);
  // cov should be non-negative
  ASSERT(aux >= 0);

  tensor_free(x);
  tensor_free(out);
  moe_free(moe);
}

// determinism test
TEST(test_determinism) {
  rng_seed(42);

  TransformerConfig cfg = {.n_layers = 1,
                           .d_model = 16,
                           .d_ff = 32,
                           .n_heads = 2,
                           .dk = 8,
                           .dv = 8,
                           .d_kv_comp = 8,
                           .max_seq_len = 16,
                           .vocab_size = 32,
                           .n_experts = 4,
                           .top_k = 2,
                           .aux_loss_coef = 0.01f};

  Transformer *t1 = transformer_create(&cfg);
  int tokens[] = {1, 2, 3, 4};
  transformer_forward(t1, tokens, 4);
  float out1[4];
  for (int i = 0; i < 4; i++)
    out1[i] = t1->logits->data[i];
  transformer_free(t1);

  // reset and run again
  rng_seed(42);
  Transformer *t2 = transformer_create(&cfg);
  transformer_forward(t2, tokens, 4);

  for (int i = 0; i < 4; i++) {
    ASSERT_NEAR(out1[i], t2->logits->data[i], 1e-6f);
  }

  transformer_free(t2);
}

// lr schedule test
TEST(test_lr_schedule) {
  LRSchedule *sched = lr_schedule_create(1e-3f, 100, 1000);

  // warmup: linear increase
  float lr_0 = lr_schedule_get(sched, 0);
  float lr_50 = lr_schedule_get(sched, 50);
  float lr_99 = lr_schedule_get(sched, 99);

  ASSERT(lr_0 < lr_50);
  ASSERT(lr_50 < lr_99);
  ASSERT_NEAR(lr_99, 1e-3f, 1e-5f);

  // decay: cosine decrease
  float lr_100 = lr_schedule_get(sched, 100);
  float lr_500 = lr_schedule_get(sched, 500);
  float lr_999 = lr_schedule_get(sched, 999);

  ASSERT(lr_100 > lr_500);
  ASSERT(lr_500 > lr_999);
  ASSERT_NEAR(lr_999, 1e-4f, 1e-5f); // min = 10% of peak

  lr_schedule_free(sched);
}

// gradient clipping test
TEST(test_gradient_clipping) {
  TransformerConfig cfg = {.n_layers = 1,
                           .d_model = 8,
                           .d_ff = 16,
                           .n_heads = 2,
                           .dk = 4,
                           .dv = 4,
                           .d_kv_comp = 4,
                           .max_seq_len = 8,
                           .vocab_size = 16,
                           .n_experts = 2,
                           .top_k = 1,
                           .aux_loss_coef = 0.01f};

  Transformer *t = transformer_create(&cfg);

  // set large gradients
  tensor_alloc_grad(t->embedding->weight);
  for (int i = 0; i < t->embedding->weight->size; i++) {
    t->embedding->weight->grad[i] = 10.0f;
  }

  float norm_before = compute_grad_norm(t);
  ASSERT(norm_before > 1.0f);

  clip_gradients(t, 1.0f);
  float norm_after = compute_grad_norm(t);

  ASSERT_NEAR(norm_after, 1.0f, 1e-5f);

  transformer_free(t);
}

// adam test
TEST(test_adam_update) {
  AdamW *opt = adamw_create(4, 0.9f, 0.999f, 1e-8f, 0.01f);
  float params[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float grads[] = {0.1f, 0.2f, 0.3f, 0.4f};
  float before[4];
  memcpy(before, params, sizeof(before));

  adamw_update(opt, params, grads, 4, 0.01f);

  // params should have changed
  int changed = 0;
  for (int i = 0; i < 4; i++) {
    if (fabsf(params[i] - before[i]) > 1e-9f)
      changed++;
  }
  ASSERT(changed == 4);

  // params should have decreased (positive gradients)
  for (int i = 0; i < 4; i++) {
    ASSERT(params[i] < before[i]);
  }

  adamw_free(opt);
}

// checkpoint test
TEST(test_checkpoint) {
  TransformerConfig cfg = {.n_layers = 1,
                           .d_model = 8,
                           .d_ff = 16,
                           .n_heads = 2,
                           .dk = 4,
                           .dv = 4,
                           .d_kv_comp = 4,
                           .max_seq_len = 8,
                           .vocab_size = 16,
                           .n_experts = 2,
                           .top_k = 1,
                           .aux_loss_coef = 0.01f};

  rng_seed(123);
  Transformer *t1 = transformer_create(&cfg);
  int param_count = count_transformer_params(t1);
  AdamW *opt1 = adamw_create(param_count, 0.9f, 0.999f, 1e-8f, 0.01f);
  LRSchedule *sched1 = lr_schedule_create(1e-3f, 10, 100);
  sched1->current_step = 42;

  // run forward
  int tokens[] = {1, 2, 3};
  transformer_forward(t1, tokens, 3);

  // save
  ASSERT(save_checkpoint(t1, opt1, sched1, "/tmp/test_ckpt.bin") == 0);

  // create new model and load
  Transformer *t2 = transformer_create(&cfg);
  AdamW *opt2 = adamw_create(param_count, 0.9f, 0.999f, 1e-8f, 0.01f);
  LRSchedule *sched2 = lr_schedule_create(1e-3f, 10, 100);

  ASSERT(load_checkpoint(t2, opt2, sched2, "/tmp/test_ckpt.bin") == 0);
  ASSERT(sched2->current_step == 42);

  // run forward - should match
  transformer_forward(t2, tokens, 3);

  for (int i = 0; i < t1->logits->size; i++) {
    ASSERT_NEAR(t1->logits->data[i], t2->logits->data[i], 1e-6f);
  }

  transformer_free(t1);
  transformer_free(t2);
  adamw_free(opt1);
  adamw_free(opt2);
  lr_schedule_free(sched1);
  lr_schedule_free(sched2);
}

// cross entropy test
TEST(test_cross_entropy) {
  float logits[] = {1.0f, 2.0f, 3.0f, 0.5f, 1.5f, 2.5f};
  int targets[] = {2, 1};
  float loss = cross_entropy_loss(logits, targets, 2, 3);

  ASSERT(loss > 0);
  ASSERT(loss < 10);

  // gradient should sum to zero per row (softmax property)
  float d_logits[6];
  cross_entropy_backward(d_logits, logits, targets, 2, 3);

  for (int t = 0; t < 2; t++) {
    float sum = 0;
    for (int i = 0; i < 3; i++) {
      sum += d_logits[t * 3 + i];
    }
    ASSERT_NEAR(sum, 0.0f, 1e-5f);
  }
}

// precision conversion tests

TEST(test_fp16_conversion) {
  float values[] = {0.0f,     1.0f,     -1.0f,     0.5f,
                    3.14159f, 65504.0f, -65504.0f, 1e-4f};
  int n = sizeof(values) / sizeof(values[0]);

  for (int i = 0; i < n; i++) {
    fp16_t h = fp32_to_fp16(values[i]);
    float restored = fp16_to_fp32(h);

    // fp16 has limited precision, allow 0.1% error for large values
    float tolerance = fabsf(values[i]) * 0.001f + 1e-4f;
    ASSERT_NEAR(restored, values[i], tolerance);
  }
}

TEST(test_bf16_conversion) {
  float values[] = {0.0f, 1.0f, -1.0f, 0.5f, 3.14159f, 1e10f, -1e10f, 1e-10f};
  int n = sizeof(values) / sizeof(values[0]);

  for (int i = 0; i < n; i++) {
    bf16_t b = fp32_to_bf16(values[i]);
    float restored = bf16_to_fp32(b);

    // bf16 truncates lower 16 bits, allow larger error
    float tolerance = fabsf(values[i]) * 0.01f + 1e-6f;
    ASSERT_NEAR(restored, values[i], tolerance);
  }
}

TEST(test_fp8_conversion) {
  // fp8 e4m3 has range roughly [-448, 448]
  float values[] = {0.0f, 1.0f, -1.0f, 0.5f, 10.0f, 100.0f, 400.0f};
  int n = sizeof(values) / sizeof(values[0]);

  for (int i = 0; i < n; i++) {
    fp8_e4m3_t f8 = fp32_to_fp8_e4m3(values[i]);
    float restored = fp8_e4m3_to_fp32(f8);

    // fp8 has very limited precision, allow 15% error
    float tolerance = fabsf(values[i]) * 0.15f + 0.1f;
    ASSERT_NEAR(restored, values[i], tolerance);
  }
}

static int run_tests(void) {
  printf("running comprehensive test suite...\n\n");

  printf("tokenizer tests:\n");
  RUN_TEST(test_tokenizer_byte);
  RUN_TEST(test_tokenizer_special);
  RUN_TEST(test_tokenizer_roundtrip);
  RUN_TEST(test_tokenizer_empty);
  RUN_TEST(test_tokenizer_binary);

  printf("\nmatmul tests:\n");
  RUN_TEST(test_matmul_basic);
  RUN_TEST(test_matmul_transB);

  printf("\nrmsnorm tests:\n");
  RUN_TEST(test_rmsnorm_forward);
  RUN_TEST(test_rmsnorm_backward);

  printf("\nrope tests:\n");
  RUN_TEST(test_rope_rotation);
  RUN_TEST(test_rope_orthogonal);

  printf("\nswiglu tests:\n");
  RUN_TEST(test_swiglu_forward);
  RUN_TEST(test_swiglu_backward);

  printf("\nmla tests:\n");
  RUN_TEST(test_mla_forward);
  RUN_TEST(test_mla_kv_compression);

  printf("\nmoe tests:\n");
  RUN_TEST(test_moe_routing);
  RUN_TEST(test_moe_load_balance);

  printf("\nsystem tests:\n");
  RUN_TEST(test_determinism);
  RUN_TEST(test_checkpoint);

  printf("\ntraining tests:\n");
  RUN_TEST(test_lr_schedule);
  RUN_TEST(test_gradient_clipping);
  RUN_TEST(test_adam_update);
  RUN_TEST(test_cross_entropy);

  printf("\nprecision tests:\n");
  RUN_TEST(test_fp16_conversion);
  RUN_TEST(test_bf16_conversion);
  RUN_TEST(test_fp8_conversion);

  printf("\n========================================\n");
  printf("results: %d/%d tests passed\n", tests_passed, tests_run);

  return (tests_passed == tests_run) ? 0 : 1;
}
