#include "shard.h"
#include "memmap.h"
#include "precision.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int ensure_dir(const char *path) {
  struct stat st;
  if (stat(path, &st) == 0) {
    return S_ISDIR(st.st_mode) ? 0 : -1;
  }
  return mkdir(path, 0755);
}

int save_sharded_checkpoint(Transformer *t, AdamW *opt, LRSchedule *sched,
                            const char *dir, int num_shards) {
  if (ensure_dir(dir) != 0) {
    fprintf(stderr, "failed to create shard directory: %s\n", dir);
    return -1;
  }

  // flatten parameters
  int param_count;
  float *params = flatten_transformer_params(t, &param_count);

  // calculate params per shard
  int64_t params_per_shard = (param_count + num_shards - 1) / num_shards;

  // write master index
  char index_path[512];
  snprintf(index_path, sizeof(index_path), "%s/index.bin", dir);

  FILE *f_idx = fopen(index_path, "wb");
  if (!f_idx) {
    free(params);
    return -1;
  }

  ShardMasterHeader master = {.magic = SHARD_MAGIC,
                              .version = SHARD_VERSION,
                              .n_shards = num_shards,
                              .dtype = DTYPE_FP32,
                              .total_params = param_count,
                              .config = t->config};
  fwrite(&master, sizeof(ShardMasterHeader), 1, f_idx);

  // write shard offsets
  for (int i = 0; i < num_shards; i++) {
    int64_t offset = (int64_t)i * params_per_shard;
    int64_t count = params_per_shard;
    if (offset + count > param_count) {
      count = param_count - offset;
    }
    fwrite(&offset, sizeof(int64_t), 1, f_idx);
    fwrite(&count, sizeof(int64_t), 1, f_idx);
  }
  fclose(f_idx);

  // write each shard
  for (int i = 0; i < num_shards; i++) {
    int64_t offset = (int64_t)i * params_per_shard;
    int64_t count = params_per_shard;
    if (offset + count > param_count) {
      count = param_count - offset;
    }

    char shard_path[512];
    snprintf(shard_path, sizeof(shard_path), "%s/shard_%04d.bin", dir, i);

    FILE *f = fopen(shard_path, "wb");
    if (!f) {
      free(params);
      return -1;
    }

    ShardFileHeader hdr = {.magic = SHARD_MAGIC,
                           .version = SHARD_VERSION,
                           .shard_id = i,
                           .total_shards = num_shards,
                           .param_offset = offset,
                           .param_count = count,
                           .dtype = DTYPE_FP32,
                           .scale = 1.0f};
    fwrite(&hdr, sizeof(ShardFileHeader), 1, f);
    fwrite(params + offset, sizeof(float), (size_t)count, f);
    fclose(f);
  }

  free(params);

  // save optimizer state
  if (opt) {
    char opt_path[512];
    snprintf(opt_path, sizeof(opt_path), "%s/optimizer.bin", dir);
    FILE *f_opt = fopen(opt_path, "wb");
    if (f_opt) {
      adamw_save(opt, f_opt);
      fclose(f_opt);
    }
  }

  // save lr schedule
  if (sched) {
    char sched_path[512];
    snprintf(sched_path, sizeof(sched_path), "%s/schedule.bin", dir);
    FILE *f_sched = fopen(sched_path, "wb");
    if (f_sched) {
      fwrite(&sched->peak_lr, sizeof(float), 1, f_sched);
      fwrite(&sched->min_lr, sizeof(float), 1, f_sched);
      fwrite(&sched->warmup_steps, sizeof(int), 1, f_sched);
      fwrite(&sched->total_steps, sizeof(int), 1, f_sched);
      fwrite(&sched->current_step, sizeof(int), 1, f_sched);
      fclose(f_sched);
    }
  }

  // save rng state
  extern uint64_t rng_state[4];
  char rng_path[512];
  snprintf(rng_path, sizeof(rng_path), "%s/rng.bin", dir);
  FILE *f_rng = fopen(rng_path, "wb");
  if (f_rng) {
    fwrite(rng_state, sizeof(uint64_t), 4, f_rng);
    fclose(f_rng);
  }

  return 0;
}

int load_sharded_checkpoint(Transformer *t, AdamW *opt, LRSchedule *sched,
                            const char *dir) {
  // read master index
  char index_path[512];
  snprintf(index_path, sizeof(index_path), "%s/index.bin", dir);

  FILE *f_idx = fopen(index_path, "rb");
  if (!f_idx)
    return -1;

  ShardMasterHeader master;
  if (fread(&master, sizeof(ShardMasterHeader), 1, f_idx) != 1) {
    fclose(f_idx);
    return -1;
  }

  if (master.magic != SHARD_MAGIC || master.version != SHARD_VERSION) {
    fclose(f_idx);
    return -2;
  }

  // verify config matches
  if (master.config.vocab_size != t->config.vocab_size ||
      master.config.d_model != t->config.d_model ||
      master.config.n_layers != t->config.n_layers) {
    fclose(f_idx);
    return -3;
  }

  // allocate full param buffer
  float *params =
      (float *)safe_malloc((size_t)master.total_params * sizeof(float));

  // read shard offsets and load each shard
  for (int i = 0; i < master.n_shards; i++) {
    int64_t offset, count;
    if (fread(&offset, sizeof(int64_t), 1, f_idx) != 1 ||
        fread(&count, sizeof(int64_t), 1, f_idx) != 1) {
      free(params);
      fclose(f_idx);
      return -1;
    }

    char shard_path[512];
    snprintf(shard_path, sizeof(shard_path), "%s/shard_%04d.bin", dir, i);

    FILE *f = fopen(shard_path, "rb");
    if (!f) {
      free(params);
      fclose(f_idx);
      return -1;
    }

    ShardFileHeader hdr;
    if (fread(&hdr, sizeof(ShardFileHeader), 1, f) != 1) {
      fclose(f);
      free(params);
      fclose(f_idx);
      return -1;
    }

    if (fread(params + offset, sizeof(float), (size_t)count, f) !=
        (size_t)count) {
      fclose(f);
      free(params);
      fclose(f_idx);
      return -1;
    }
    fclose(f);
  }
  fclose(f_idx);

  unflatten_transformer_params(t, params);
  free(params);

  // load optimizer
  if (opt) {
    char opt_path[512];
    snprintf(opt_path, sizeof(opt_path), "%s/optimizer.bin", dir);
    FILE *f_opt = fopen(opt_path, "rb");
    if (f_opt) {
      adamw_load(opt, f_opt);
      fclose(f_opt);
    }
  }

  // load schedule
  if (sched) {
    char sched_path[512];
    snprintf(sched_path, sizeof(sched_path), "%s/schedule.bin", dir);
    FILE *f_sched = fopen(sched_path, "rb");
    if (f_sched) {
      fread(&sched->peak_lr, sizeof(float), 1, f_sched);
      fread(&sched->min_lr, sizeof(float), 1, f_sched);
      fread(&sched->warmup_steps, sizeof(int), 1, f_sched);
      fread(&sched->total_steps, sizeof(int), 1, f_sched);
      fread(&sched->current_step, sizeof(int), 1, f_sched);
      fclose(f_sched);
    }
  }

  // load rng state
  extern uint64_t rng_state[4];
  char rng_path[512];
  snprintf(rng_path, sizeof(rng_path), "%s/rng.bin", dir);
  FILE *f_rng = fopen(rng_path, "rb");
  if (f_rng) {
    fread(rng_state, sizeof(uint64_t), 4, f_rng);
    fclose(f_rng);
  }

  return 0;
}

int save_optimizer_sharded(AdamW *opt, const char *dir, int num_shards) {
  if (ensure_dir(dir) != 0)
    return -1;

  int64_t per_shard = (opt->size + num_shards - 1) / num_shards;

  for (int i = 0; i < num_shards; i++) {
    int64_t offset = (int64_t)i * per_shard;
    int64_t count = per_shard;
    if (offset + count > opt->size)
      count = opt->size - offset;
    if (count <= 0)
      break;

    char path[512];
    snprintf(path, sizeof(path), "%s/opt_shard_%04d.bin", dir, i);

    FILE *f = fopen(path, "wb");
    if (!f)
      return -1;

    fwrite(&offset, sizeof(int64_t), 1, f);
    fwrite(&count, sizeof(int64_t), 1, f);
    fwrite(opt->m + offset, sizeof(float), (size_t)count, f);
    fwrite(opt->v + offset, sizeof(float), (size_t)count, f);
    fclose(f);
  }

  // write optimizer metadata
  char meta_path[512];
  snprintf(meta_path, sizeof(meta_path), "%s/opt_meta.bin", dir);
  FILE *f = fopen(meta_path, "wb");
  if (!f)
    return -1;

  fwrite(&opt->size, sizeof(int), 1, f);
  fwrite(&opt->beta1, sizeof(float), 1, f);
  fwrite(&opt->beta2, sizeof(float), 1, f);
  fwrite(&opt->eps, sizeof(float), 1, f);
  fwrite(&opt->weight_decay, sizeof(float), 1, f);
  fwrite(&opt->t, sizeof(int), 1, f);
  fwrite(&num_shards, sizeof(int), 1, f);
  fclose(f);

  return 0;
}

int load_optimizer_sharded(AdamW *opt, const char *dir) {
  char meta_path[512];
  snprintf(meta_path, sizeof(meta_path), "%s/opt_meta.bin", dir);
  FILE *f = fopen(meta_path, "rb");
  if (!f)
    return -1;

  int saved_size, num_shards;
  if (fread(&saved_size, sizeof(int), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (saved_size != opt->size) {
    fclose(f);
    return -2;
  }

  fread(&opt->beta1, sizeof(float), 1, f);
  fread(&opt->beta2, sizeof(float), 1, f);
  fread(&opt->eps, sizeof(float), 1, f);
  fread(&opt->weight_decay, sizeof(float), 1, f);
  fread(&opt->t, sizeof(int), 1, f);
  fread(&num_shards, sizeof(int), 1, f);
  fclose(f);

  for (int i = 0; i < num_shards; i++) {
    char path[512];
    snprintf(path, sizeof(path), "%s/opt_shard_%04d.bin", dir, i);

    FILE *fs = fopen(path, "rb");
    if (!fs)
      continue;

    int64_t offset, count;
    fread(&offset, sizeof(int64_t), 1, fs);
    fread(&count, sizeof(int64_t), 1, fs);
    fread(opt->m + offset, sizeof(float), (size_t)count, fs);
    fread(opt->v + offset, sizeof(float), (size_t)count, fs);
    fclose(fs);
  }

  return 0;
}

// dirty tracking for incremental saves

DirtyTracker *dirty_tracker_create(int64_t param_count, int block_size) {
  DirtyTracker *dt = (DirtyTracker *)safe_malloc(sizeof(DirtyTracker));
  dt->param_count = param_count;
  dt->block_size = block_size;

  int64_t n_blocks = (param_count + block_size - 1) / block_size;
  int64_t mask_size = (n_blocks + 63) / 64;
  dt->dirty_mask = (uint64_t *)safe_calloc((size_t)mask_size, sizeof(uint64_t));

  return dt;
}

void dirty_tracker_free(DirtyTracker *dt) {
  if (dt) {
    free(dt->dirty_mask);
    free(dt);
  }
}

void dirty_tracker_mark(DirtyTracker *dt, int64_t offset, int64_t count) {
  int64_t start_block = offset / dt->block_size;
  int64_t end_block = (offset + count - 1) / dt->block_size;

  for (int64_t b = start_block; b <= end_block; b++) {
    dt->dirty_mask[b / 64] |= (1ULL << (b % 64));
  }
}

void dirty_tracker_clear(DirtyTracker *dt) {
  int64_t n_blocks = (dt->param_count + dt->block_size - 1) / dt->block_size;
  int64_t mask_size = (n_blocks + 63) / 64;
  memset(dt->dirty_mask, 0, (size_t)mask_size * sizeof(uint64_t));
}

int save_incremental_checkpoint(Transformer *t, DirtyTracker *dt,
                                const char *path) {
  int param_count;
  float *params = flatten_transformer_params(t, &param_count);

  FILE *f = fopen(path, "wb");
  if (!f) {
    free(params);
    return -1;
  }

  // write header
  int magic = SHARD_MAGIC;
  int version = SHARD_VERSION;
  fwrite(&magic, sizeof(int), 1, f);
  fwrite(&version, sizeof(int), 1, f);
  fwrite(&param_count, sizeof(int), 1, f);
  fwrite(&dt->block_size, sizeof(int), 1, f);

  // write dirty blocks
  int64_t n_blocks = (param_count + dt->block_size - 1) / dt->block_size;
  int written_blocks = 0;

  for (int64_t b = 0; b < n_blocks; b++) {
    if (dt->dirty_mask[b / 64] & (1ULL << (b % 64))) {
      int64_t offset = b * dt->block_size;
      int64_t count = dt->block_size;
      if (offset + count > param_count)
        count = param_count - offset;

      fwrite(&offset, sizeof(int64_t), 1, f);
      fwrite(&count, sizeof(int64_t), 1, f);
      fwrite(params + offset, sizeof(float), (size_t)count, f);
      written_blocks++;
    }
  }

  fclose(f);
  free(params);

  return written_blocks;
}
