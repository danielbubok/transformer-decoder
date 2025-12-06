#ifndef MEMMAP_H
#define MEMMAP_H

#include <stddef.h>
#include <stdint.h>

// memory-mapped file handle for streaming large weights
typedef struct {
  int fd;
  void *base;
  size_t size;
  size_t mapped_offset;
  size_t mapped_size;
  int is_write;
} MMapHandle;

// shard metadata
typedef struct {
  int shard_id;
  int total_shards;
  int64_t param_offset;
  int64_t param_count;
  char path[256];
} ShardInfo;

// shard index for large models
typedef struct {
  ShardInfo *shards;
  int n_shards;
  int64_t total_params;
} ShardIndex;

// open file for memory mapping
MMapHandle *mmap_open_read(const char *path);
MMapHandle *mmap_open_write(const char *path, size_t size);

// get pointer to slice of mapped file
// automatically remaps if needed
void *mmap_get_slice(MMapHandle *h, size_t offset, size_t len);

// sync written data to disk
void mmap_sync(MMapHandle *h);

// close handle
void mmap_close(MMapHandle *h);

// streaming weight loader for 100b+ models
typedef struct {
  ShardIndex index;
  MMapHandle *current_shard;
  int current_shard_id;
} WeightStreamer;

// create streamer from shard directory
WeightStreamer *weight_streamer_create(const char *shard_dir);

// get weights for parameter range
// returns pointer to fp32 data (may convert on the fly)
float *weight_streamer_get(WeightStreamer *ws, int64_t offset, int64_t count);

// free streamer
void weight_streamer_free(WeightStreamer *ws);

// utility: estimate memory for model config
typedef struct {
  int64_t param_bytes;
  int64_t activation_bytes;
  int64_t gradient_bytes;
  int64_t optimizer_bytes;
  int64_t total_bytes;
} MemoryEstimate;

MemoryEstimate estimate_memory(int n_layers, int d_model, int d_ff, int n_heads,
                               int n_experts, int top_k, int vocab_size,
                               int max_seq_len, int batch_size,
                               int precision_bytes);

// print memory estimate in human readable form
void print_memory_estimate(MemoryEstimate *est);

#endif
