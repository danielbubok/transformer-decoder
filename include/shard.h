#ifndef SHARD_H
#define SHARD_H

#include "train.h"
#include "transformer.h"
#include <stdint.h>

#define SHARD_MAGIC 0x53485244 // "SHRD"
#define SHARD_VERSION 1

// header for each shard file
typedef struct {
  int magic;
  int version;
  int shard_id;
  int total_shards;
  int64_t param_offset;
  int64_t param_count;
  int dtype;   // 0=fp32, 1=fp16, 2=bf16
  float scale; // for quantized formats
} ShardFileHeader;

// master index file structure
typedef struct {
  int magic;
  int version;
  int n_shards;
  int dtype;
  int64_t total_params;
  TransformerConfig config;
} ShardMasterHeader;

// save model to sharded checkpoint format
// creates directory with:
//   index.bin - master index
//   shard_0000.bin ... shard_NNNN.bin - weight shards
int save_sharded_checkpoint(Transformer *t, AdamW *opt, LRSchedule *sched,
                            const char *dir, int num_shards);

// load model from sharded checkpoint
// supports streaming load for large models
int load_sharded_checkpoint(Transformer *t, AdamW *opt, LRSchedule *sched,
                            const char *dir);

// save optimizer state separately (can be much larger than weights)
int save_optimizer_sharded(AdamW *opt, const char *dir, int num_shards);
int load_optimizer_sharded(AdamW *opt, const char *dir);

// incremental checkpoint: save only changed parameters
// uses dirty tracking bitmask
typedef struct {
  uint64_t *dirty_mask;
  int64_t param_count;
  int block_size;
} DirtyTracker;

DirtyTracker *dirty_tracker_create(int64_t param_count, int block_size);
void dirty_tracker_free(DirtyTracker *dt);
void dirty_tracker_mark(DirtyTracker *dt, int64_t offset, int64_t count);
void dirty_tracker_clear(DirtyTracker *dt);
int save_incremental_checkpoint(Transformer *t, DirtyTracker *dt,
                                const char *path);

#endif
