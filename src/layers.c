#include "layers.h"
#include "ops.h"
#include <math.h>
#include <stdlib.h>

LayerNorm *layernorm_create(int dim) {
  LayerNorm *ln = (LayerNorm *)safe_malloc(sizeof(LayerNorm));
  ln->dim = dim;
  ln->gamma = tensor_create_2d(1, dim);
  ln->beta = tensor_create_2d(1, dim);
  for (int i = 0; i < dim; i++) {
    ln->gamma->data[i] = 1.0f;
    ln->beta->data[i] = 0.0f;
  }
  return ln;
}

void layernorm_free(LayerNorm *ln) {
  if (ln) {
    tensor_free(ln->gamma);
    tensor_free(ln->beta);
    free(ln);
  }
}

void layernorm_forward(LayerNorm *ln, Tensor *x) {
  int seq_len = x->shape[0];
  int dim = x->shape[1];
  float eps = 1e-6f;
  for (int t = 0; t < seq_len; t++) {
    float *row = tensor_at_2d(x, t, 0);
    float mean = 0.0f;
    for (int i = 0; i < dim; i++)
      mean += row[i];
    mean /= (float)dim;
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
      float diff = row[i] - mean;
      var += diff * diff;
    }
    var /= (float)dim;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; i++) {
      row[i] =
          ln->gamma->data[i] * (row[i] - mean) * inv_std + ln->beta->data[i];
    }
  }
}

Linear *linear_create(int in_dim, int out_dim, int use_bias) {
  Linear *l = (Linear *)safe_malloc(sizeof(Linear));
  l->in_dim = in_dim;
  l->out_dim = out_dim;
  l->weight = tensor_create_2d(in_dim, out_dim);
  tensor_init_xavier(l->weight, in_dim, out_dim);
  if (use_bias) {
    l->bias = tensor_create_2d(1, out_dim);
    tensor_zero(l->bias);
  } else {
    l->bias = NULL;
  }
  return l;
}

void linear_free(Linear *l) {
  if (l) {
    tensor_free(l->weight);
    tensor_free(l->bias);
    free(l);
  }
}

void linear_forward(Linear *l, Tensor *y, Tensor *x) {
  int seq_len = x->shape[0];
  matmul(y->data, x->data, l->weight->data, seq_len, l->in_dim, l->out_dim);
  if (l->bias) {
    for (int t = 0; t < seq_len; t++) {
      for (int i = 0; i < l->out_dim; i++) {
        *tensor_at_2d(y, t, i) += l->bias->data[i];
      }
    }
  }
}

PositionalEncoding *positional_encoding_create(int max_len, int dim) {
  PositionalEncoding *pe =
      (PositionalEncoding *)safe_malloc(sizeof(PositionalEncoding));
  pe->max_len = max_len;
  pe->dim = dim;
  pe->encoding = tensor_create_2d(max_len, dim);
  for (int pos = 0; pos < max_len; pos++) {
    for (int i = 0; i < dim; i++) {
      float angle =
          (float)pos / powf(10000.0f, (float)((i / 2) * 2) / (float)dim);
      if (i % 2 == 0) {
        *tensor_at_2d(pe->encoding, pos, i) = sinf(angle);
      } else {
        *tensor_at_2d(pe->encoding, pos, i) = cosf(angle);
      }
    }
  }
  return pe;
}

void positional_encoding_free(PositionalEncoding *pe) {
  if (pe) {
    tensor_free(pe->encoding);
    free(pe);
  }
}

void positional_encoding_add(PositionalEncoding *pe, Tensor *x) {
  int seq_len = x->shape[0];
  int dim = x->shape[1];
  for (int t = 0; t < seq_len; t++) {
    for (int i = 0; i < dim; i++) {
      *tensor_at_2d(x, t, i) += *tensor_at_2d(pe->encoding, t, i);
    }
  }
}

ScaledDotProductAttention *sdpa_create(int max_seq_len, int dk) {
  ScaledDotProductAttention *attn = (ScaledDotProductAttention *)safe_malloc(
      sizeof(ScaledDotProductAttention));
  attn->scores = tensor_create_2d(max_seq_len, max_seq_len);
  attn->attn = tensor_create_2d(max_seq_len, max_seq_len);
  attn->scale = 1.0f / sqrtf((float)dk);
  return attn;
}

void sdpa_free(ScaledDotProductAttention *attn) {
  if (attn) {
    tensor_free(attn->scores);
    tensor_free(attn->attn);
    free(attn);
  }
}

void sdpa_forward(ScaledDotProductAttention *attn, Tensor *output, Tensor *Q,
                  Tensor *K, Tensor *V, int *mask, int seq_len_q, int seq_len_k,
                  int dk, int dv) {
  matmul_transB(attn->scores->data, Q->data, K->data, seq_len_q, dk, seq_len_k);
  for (int i = 0; i < seq_len_q * seq_len_k; i++) {
    attn->scores->data[i] *= attn->scale;
  }
  if (mask) {
    for (int i = 0; i < seq_len_q; i++) {
      for (int j = 0; j < seq_len_k; j++) {
        if (!mask[i * seq_len_k + j]) {
          *tensor_at_2d(attn->scores, i, j) = -1e9f;
        }
      }
    }
  }
  for (int i = 0; i < seq_len_q; i++) {
    softmax(tensor_at_2d(attn->scores, i, 0), seq_len_k);
  }
  matmul(output->data, attn->scores->data, V->data, seq_len_q, seq_len_k, dv);
}

MultiHeadAttention *mha_create(int n_heads, int d_model, int dk, int dv,
                               int max_seq_len) {
  MultiHeadAttention *mha =
      (MultiHeadAttention *)safe_malloc(sizeof(MultiHeadAttention));
  mha->n_heads = n_heads;
  mha->d_model = d_model;
  mha->dk = dk;
  mha->dv = dv;
  for (int h = 0; h < n_heads; h++) {
    mha->W_Q[h] = linear_create(d_model, dk, 0);
    mha->W_K[h] = linear_create(d_model, dk, 0);
    mha->W_V[h] = linear_create(d_model, dv, 0);
    mha->Q_proj[h] = tensor_create_2d(max_seq_len, dk);
    mha->K_proj[h] = tensor_create_2d(max_seq_len, dk);
    mha->V_proj[h] = tensor_create_2d(max_seq_len, dv);
    mha->head_out[h] = tensor_create_2d(max_seq_len, dv);
  }
  mha->W_O = linear_create(n_heads * dv, d_model, 0);
  mha->concat = tensor_create_2d(max_seq_len, n_heads * dv);
  mha->sdpa = sdpa_create(max_seq_len, dk);
  return mha;
}

void mha_free(MultiHeadAttention *mha) {
  if (mha) {
    for (int h = 0; h < mha->n_heads; h++) {
      linear_free(mha->W_Q[h]);
      linear_free(mha->W_K[h]);
      linear_free(mha->W_V[h]);
      tensor_free(mha->Q_proj[h]);
      tensor_free(mha->K_proj[h]);
      tensor_free(mha->V_proj[h]);
      tensor_free(mha->head_out[h]);
    }
    linear_free(mha->W_O);
    tensor_free(mha->concat);
    sdpa_free(mha->sdpa);
    free(mha);
  }
}

void mha_forward(MultiHeadAttention *mha, Tensor *output, Tensor *Q, Tensor *K,
                 Tensor *V, int *mask, int seq_len_q, int seq_len_k) {
  for (int h = 0; h < mha->n_heads; h++) {
    linear_forward(mha->W_Q[h], mha->Q_proj[h], Q);
    linear_forward(mha->W_K[h], mha->K_proj[h], K);
    linear_forward(mha->W_V[h], mha->V_proj[h], V);
    sdpa_forward(mha->sdpa, mha->head_out[h], mha->Q_proj[h], mha->K_proj[h],
                 mha->V_proj[h], mask, seq_len_q, seq_len_k, mha->dk, mha->dv);
  }
  for (int t = 0; t < seq_len_q; t++) {
    for (int h = 0; h < mha->n_heads; h++) {
      for (int i = 0; i < mha->dv; i++) {
        *tensor_at_2d(mha->concat, t, h * mha->dv + i) =
            *tensor_at_2d(mha->head_out[h], t, i);
      }
    }
  }
  linear_forward(mha->W_O, output, mha->concat);
}

FeedForward *ff_create(int d_model, int d_ff, int max_seq_len) {
  FeedForward *ff = (FeedForward *)safe_malloc(sizeof(FeedForward));
  ff->d_model = d_model;
  ff->d_ff = d_ff;
  ff->fc1 = linear_create(d_model, d_ff, 1);
  ff->fc2 = linear_create(d_ff, d_model, 1);
  ff->hidden = tensor_create_2d(max_seq_len, d_ff);
  return ff;
}

void ff_free(FeedForward *ff) {
  if (ff) {
    linear_free(ff->fc1);
    linear_free(ff->fc2);
    tensor_free(ff->hidden);
    free(ff);
  }
}

void ff_forward(FeedForward *ff, Tensor *output, Tensor *input, int seq_len) {
  linear_forward(ff->fc1, ff->hidden, input);
  relu_inplace(ff->hidden->data, seq_len * ff->d_ff);
  linear_forward(ff->fc2, output, ff->hidden);
}

Embedding *embedding_create(int vocab_size, int d_model) {
  Embedding *emb = (Embedding *)safe_malloc(sizeof(Embedding));
  emb->vocab_size = vocab_size;
  emb->d_model = d_model;
  emb->scale = sqrtf((float)d_model);
  emb->weight = tensor_create_2d(vocab_size, d_model);
  tensor_init_xavier(emb->weight, vocab_size, d_model);
  return emb;
}

void embedding_free(Embedding *emb) {
  if (emb) {
    tensor_free(emb->weight);
    free(emb);
  }
}

void embedding_forward(Embedding *emb, Tensor *output, int *tokens,
                       int seq_len) {
  for (int t = 0; t < seq_len; t++) {
    int token = tokens[t];
    for (int i = 0; i < emb->d_model; i++) {
      *tensor_at_2d(output, t, i) =
          *tensor_at_2d(emb->weight, token, i) * emb->scale;
    }
  }
}
