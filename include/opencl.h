#ifndef OPENCL_H
#define OPENCL_H

#define CL_TARGET_OPENCL_VERSION 300

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stddef.h>

// error checking macro
#define CL_CHECK(err)                                                          \
  do {                                                                         \
    cl_int _err = (err);                                                       \
    if (_err != CL_SUCCESS) {                                                  \
      fprintf(stderr, "opencl error %d at %s:%d\n", _err, __FILE__, __LINE__); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

typedef struct {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  int is_gpu;
  int initialized;

  // kernels
  cl_kernel k_tiled_gemm;
  cl_kernel k_tiled_gemm_transB;
  cl_kernel k_softmax_stable;
  cl_kernel k_rmsnorm_forward;
  cl_kernel k_rmsnorm_backward;
  cl_kernel k_rope_forward;
  cl_kernel k_swiglu_forward;
  cl_kernel k_swiglu_backward;
  cl_kernel k_topk_routing;
  cl_kernel k_head_attention;
  cl_kernel k_elementwise_add;
  cl_kernel k_elementwise_mul;
  cl_kernel k_elementwise_scale;
  cl_kernel k_elementwise_add_scaled;
  cl_kernel k_bias_add;
  cl_kernel k_cross_entropy_forward;
  cl_kernel k_cross_entropy_backward;
  cl_kernel k_adam_update;
} CLContext;

// global context
extern CLContext g_cl;

// initialization and cleanup
int cl_init(void);
void cl_cleanup(void);
int cl_is_available(void);

// memory management
cl_mem cl_alloc(size_t size);
cl_mem cl_alloc_copy(const float *data, size_t size);
void cl_free(cl_mem buf);
void cl_copy_to_device(cl_mem buf, const float *data, size_t size);
void cl_copy_to_host(float *data, cl_mem buf, size_t size);
void cl_zero(cl_mem buf, size_t size);

// kernel launches
void cl_gemm(cl_mem A, cl_mem B, cl_mem C, int M, int K, int N);
void cl_gemm_transB(cl_mem A, cl_mem B, cl_mem C, int M, int K, int N);
void cl_softmax(cl_mem x, int rows, int cols);
void cl_rmsnorm_forward(cl_mem x, cl_mem gamma, cl_mem y, int rows, int dim,
                        float eps);
void cl_rope_forward(cl_mem x, int seq_len, int dim, float base);
void cl_swiglu_forward(cl_mem gate, cl_mem value, cl_mem out, int size);
void cl_swiglu_backward(cl_mem dout, cl_mem gate, cl_mem value, cl_mem dgate,
                        cl_mem dvalue, int size);
void cl_topk_routing(cl_mem logits, cl_mem indices, cl_mem weights, int seq_len,
                     int n_experts, int top_k);
void cl_head_attention(cl_mem Q, cl_mem K, cl_mem V, cl_mem out, cl_mem mask,
                       int seq_q, int seq_k, int dk, int dv, float scale);
void cl_elementwise_add(cl_mem a, cl_mem b, cl_mem c, int size);
void cl_elementwise_scale(cl_mem x, float scale, int size);
void cl_bias_add(cl_mem y, cl_mem bias, int rows, int cols);
void cl_cross_entropy_forward(cl_mem logits, cl_mem targets, cl_mem losses,
                              int seq_len, int vocab_size);
void cl_cross_entropy_backward(cl_mem logits, cl_mem targets, cl_mem dlogits,
                               int seq_len, int vocab_size);
void cl_adam_update(cl_mem params, cl_mem grads, cl_mem m, cl_mem v, float lr,
                    float beta1, float beta2, float eps, float weight_decay,
                    float bc1, float bc2, int size);

// synchronization
void cl_finish(void);

#endif
