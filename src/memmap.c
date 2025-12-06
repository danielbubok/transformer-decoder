#define _XOPEN_SOURCE 700
#include "memmap.h"
#include "tensor.h"
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// page size for alignment
#define PAGE_SIZE 4096
#define ALIGN_DOWN(x, a) ((x) & ~((a) - 1))

MMapHandle *mmap_open_read(const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0)
    return NULL;

  struct stat st;
  if (fstat(fd, &st) < 0) {
    close(fd);
    return NULL;
  }

  MMapHandle *h = (MMapHandle *)safe_malloc(sizeof(MMapHandle));
  h->fd = fd;
  h->size = (size_t)st.st_size;
  h->is_write = 0;
  h->mapped_offset = 0;
  h->mapped_size = 0;
  h->base = NULL;

  // map entire file if small enough
  if (h->size <= (size_t)1024 * 1024 * 1024) {
    h->base = mmap(NULL, h->size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (h->base == MAP_FAILED) {
      h->base = NULL;
    } else {
      h->mapped_size = h->size;
    }
  }

  return h;
}

MMapHandle *mmap_open_write(const char *path, size_t size) {
  int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0)
    return NULL;

  if (ftruncate(fd, (off_t)size) < 0) {
    close(fd);
    return NULL;
  }

  MMapHandle *h = (MMapHandle *)safe_malloc(sizeof(MMapHandle));
  h->fd = fd;
  h->size = size;
  h->is_write = 1;
  h->mapped_offset = 0;
  h->mapped_size = 0;
  h->base = NULL;

  // map entire file if small enough
  if (size <= (size_t)1024 * 1024 * 1024) {
    h->base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (h->base == MAP_FAILED) {
      h->base = NULL;
    } else {
      h->mapped_size = size;
    }
  }

  return h;
}

void *mmap_get_slice(MMapHandle *h, size_t offset, size_t len) {
  if (!h || offset + len > h->size)
    return NULL;

  // if already fully mapped
  if (h->base && offset >= h->mapped_offset &&
      offset + len <= h->mapped_offset + h->mapped_size) {
    return (char *)h->base + (offset - h->mapped_offset);
  }

  // need to remap
  if (h->base) {
    munmap(h->base, h->mapped_size);
    h->base = NULL;
  }

  // align to page boundary
  size_t aligned_offset = ALIGN_DOWN(offset, PAGE_SIZE);
  size_t extra = offset - aligned_offset;
  size_t map_size = len + extra;

  // map at least 256mb chunks for efficiency
  if (map_size < 256 * 1024 * 1024) {
    map_size = 256 * 1024 * 1024;
  }
  if (aligned_offset + map_size > h->size) {
    map_size = h->size - aligned_offset;
  }

  int prot = h->is_write ? (PROT_READ | PROT_WRITE) : PROT_READ;
  int flags = h->is_write ? MAP_SHARED : MAP_PRIVATE;

  h->base = mmap(NULL, map_size, prot, flags, h->fd, (off_t)aligned_offset);
  if (h->base == MAP_FAILED) {
    h->base = NULL;
    return NULL;
  }

  h->mapped_offset = aligned_offset;
  h->mapped_size = map_size;

  return (char *)h->base + extra;
}

void mmap_sync(MMapHandle *h) {
  if (h && h->base && h->is_write) {
    msync(h->base, h->mapped_size, MS_SYNC);
  }
}

void mmap_close(MMapHandle *h) {
  if (h) {
    if (h->base) {
      if (h->is_write) {
        msync(h->base, h->mapped_size, MS_SYNC);
      }
      munmap(h->base, h->mapped_size);
    }
    close(h->fd);
    free(h);
  }
}

// shard index functions

static int load_shard_index(ShardIndex *idx, const char *shard_dir) {
  char index_path[512];
  snprintf(index_path, sizeof(index_path), "%s/index.bin", shard_dir);

  FILE *f = fopen(index_path, "rb");
  if (!f)
    return -1;

  if (fread(&idx->n_shards, sizeof(int), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (fread(&idx->total_params, sizeof(int64_t), 1, f) != 1) {
    fclose(f);
    return -1;
  }

  idx->shards =
      (ShardInfo *)safe_malloc((size_t)idx->n_shards * sizeof(ShardInfo));

  for (int i = 0; i < idx->n_shards; i++) {
    if (fread(&idx->shards[i].shard_id, sizeof(int), 1, f) != 1 ||
        fread(&idx->shards[i].total_shards, sizeof(int), 1, f) != 1 ||
        fread(&idx->shards[i].param_offset, sizeof(int64_t), 1, f) != 1 ||
        fread(&idx->shards[i].param_count, sizeof(int64_t), 1, f) != 1) {
      free(idx->shards);
      fclose(f);
      return -1;
    }
    snprintf(idx->shards[i].path, sizeof(idx->shards[i].path),
             "%s/shard_%04d.bin", shard_dir, i);
  }

  fclose(f);
  return 0;
}

WeightStreamer *weight_streamer_create(const char *shard_dir) {
  WeightStreamer *ws = (WeightStreamer *)safe_malloc(sizeof(WeightStreamer));
  ws->current_shard = NULL;
  ws->current_shard_id = -1;

  if (load_shard_index(&ws->index, shard_dir) != 0) {
    free(ws);
    return NULL;
  }

  return ws;
}

float *weight_streamer_get(WeightStreamer *ws, int64_t offset, int64_t count) {
  if (!ws)
    return NULL;

  // find shard containing offset
  int shard_id = -1;
  for (int i = 0; i < ws->index.n_shards; i++) {
    if (offset >= ws->index.shards[i].param_offset &&
        offset < ws->index.shards[i].param_offset +
                     ws->index.shards[i].param_count) {
      shard_id = i;
      break;
    }
  }

  if (shard_id < 0)
    return NULL;

  // open shard if needed
  if (ws->current_shard_id != shard_id) {
    if (ws->current_shard) {
      mmap_close(ws->current_shard);
    }
    ws->current_shard = mmap_open_read(ws->index.shards[shard_id].path);
    ws->current_shard_id = shard_id;
  }

  if (!ws->current_shard)
    return NULL;

  int64_t local_offset = offset - ws->index.shards[shard_id].param_offset;
  return (float *)mmap_get_slice(ws->current_shard,
                                 (size_t)(local_offset * sizeof(float)),
                                 (size_t)(count * sizeof(float)));
}

void weight_streamer_free(WeightStreamer *ws) {
  if (ws) {
    if (ws->current_shard) {
      mmap_close(ws->current_shard);
    }
    free(ws->index.shards);
    free(ws);
  }
}

// memory estimation

MemoryEstimate estimate_memory(int n_layers, int d_model, int d_ff, int n_heads,
                               int n_experts, int top_k, int vocab_size,
                               int max_seq_len, int batch_size,
                               int precision_bytes) {
  MemoryEstimate est = {0};

  // embedding: vocab_size * d_model
  int64_t emb_params = (int64_t)vocab_size * d_model;

  // per-layer params:
  // mla: w_q (d_model * n_heads * dk), w_kv_down (d_model * d_kv_comp)
  //      w_k_up, w_v_up per head, w_o
  // use n_heads for more accurate calculation
  int64_t attn_params =
      (int64_t)n_heads * 4 * (d_model / n_heads) * (d_model / n_heads) +
      2LL * d_model * d_model;

  // moe: router (d_model * n_experts)
  //      per expert: 3 * d_model * d_ff (gate, value, down projections)
  int64_t moe_params =
      (int64_t)d_model * n_experts + (int64_t)n_experts * 3 * d_model * d_ff;

  // norms: 2 * d_model per layer
  int64_t norm_params = 2LL * d_model;

  int64_t layer_params = attn_params + moe_params + norm_params;
  int64_t total_params = emb_params + n_layers * layer_params +
                         (int64_t)d_model * vocab_size; // output proj

  est.param_bytes = total_params * precision_bytes;

  // activations per batch
  // embedding output: batch * seq * d_model
  // attention intermediate: batch * seq * d_model * ~4
  // moe: batch * seq * d_model (only top_k active experts)
  int64_t act_per_layer =
      (int64_t)(4 + top_k) * batch_size * max_seq_len * d_model;
  est.activation_bytes = n_layers * act_per_layer * precision_bytes;

  // gradients (same as params for master weights)
  est.gradient_bytes = total_params * 4; // always fp32

  // optimizer state: 2x for adam m and v
  est.optimizer_bytes = total_params * 4 * 2;

  est.total_bytes = est.param_bytes + est.activation_bytes +
                    est.gradient_bytes + est.optimizer_bytes;

  return est;
}

void print_memory_estimate(MemoryEstimate *est) {
  printf("memory estimate:\n");
  printf("  parameters:   %8.2f gb\n", (double)est->param_bytes / 1e9);
  printf("  activations:  %8.2f gb\n", (double)est->activation_bytes / 1e9);
  printf("  gradients:    %8.2f gb\n", (double)est->gradient_bytes / 1e9);
  printf("  optimizer:    %8.2f gb\n", (double)est->optimizer_bytes / 1e9);
  printf("  total:        %8.2f gb\n", (double)est->total_bytes / 1e9);
}
