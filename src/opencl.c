#include "opencl.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CLContext g_cl = {0};

static char *read_kernel_source(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    // try alternate paths
    const char *alt_paths[] = {"kernels/kernels.cl", "../kernels/kernels.cl",
                               "./kernels.cl"};
    for (size_t i = 0; i < sizeof(alt_paths) / sizeof(alt_paths[0]); i++) {
      f = fopen(alt_paths[i], "rb");
      if (f)
        break;
    }
  }
  if (!f) {
    fprintf(stderr, "failed to open kernel file: %s\n", path);
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  *len = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  char *src = (char *)malloc(*len + 1);
  if (fread(src, 1, *len, f) != *len) {
    free(src);
    fclose(f);
    return NULL;
  }
  src[*len] = '\0';
  fclose(f);
  return src;
}

int cl_init(void) {
  if (g_cl.initialized)
    return 1;

  cl_int err;
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    fprintf(stderr, "no opencl platforms found, using cpu fallback\n");
    return 0;
  }

  cl_platform_id *platforms =
      (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
  clGetPlatformIDs(num_platforms, platforms, NULL);

  // try to find a gpu
  g_cl.is_gpu = 0;
  for (cl_uint p = 0; p < num_platforms && !g_cl.is_gpu; p++) {
    cl_uint num_devices;
    err =
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (err == CL_SUCCESS && num_devices > 0) {
      g_cl.platform = platforms[p];
      clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 1, &g_cl.device, NULL);
      g_cl.is_gpu = 1;
    }
  }

  // fallback to cpu
  if (!g_cl.is_gpu) {
    for (cl_uint p = 0; p < num_platforms; p++) {
      cl_uint num_devices;
      err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_CPU, 0, NULL,
                           &num_devices);
      if (err == CL_SUCCESS && num_devices > 0) {
        g_cl.platform = platforms[p];
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_CPU, 1, &g_cl.device, NULL);
        break;
      }
    }
  }

  free(platforms);

  if (!g_cl.device) {
    fprintf(stderr, "no opencl device found, using cpu fallback\n");
    return 0;
  }

  // print device info
  char device_name[256];
  clGetDeviceInfo(g_cl.device, CL_DEVICE_NAME, sizeof(device_name), device_name,
                  NULL);
  printf("opencl device: %s (%s)\n", device_name, g_cl.is_gpu ? "gpu" : "cpu");

  // create context
  g_cl.context = clCreateContext(NULL, 1, &g_cl.device, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create opencl context\n");
    return 0;
  }

  // create command queue
  // use clCreateCommandQueueWithProperties for OpenCL 2.0+
  cl_queue_properties properties[] = {0};
  g_cl.queue = clCreateCommandQueueWithProperties(g_cl.context, g_cl.device,
                                                  properties, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create command queue\n");
    clReleaseContext(g_cl.context);
    return 0;
  }

  // load and compile kernels
  size_t src_len;
  char *src = read_kernel_source("kernels/kernels.cl", &src_len);
  if (!src) {
    fprintf(stderr, "failed to load kernel source\n");
    clReleaseCommandQueue(g_cl.queue);
    clReleaseContext(g_cl.context);
    return 0;
  }

  const char *src_ptr = src;
  g_cl.program =
      clCreateProgramWithSource(g_cl.context, 1, &src_ptr, &src_len, &err);
  free(src);

  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create program\n");
    clReleaseCommandQueue(g_cl.queue);
    clReleaseContext(g_cl.context);
    return 0;
  }

  err = clBuildProgram(g_cl.program, 1, &g_cl.device, "-cl-fast-relaxed-math",
                       NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(g_cl.program, g_cl.device, CL_PROGRAM_BUILD_LOG, 0,
                          NULL, &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo(g_cl.program, g_cl.device, CL_PROGRAM_BUILD_LOG,
                          log_size, log, NULL);
    fprintf(stderr, "kernel build failed:\n%s\n", log);
    free(log);
    clReleaseProgram(g_cl.program);
    clReleaseCommandQueue(g_cl.queue);
    clReleaseContext(g_cl.context);
    return 0;
  }

  // create kernel objects
  g_cl.k_tiled_gemm = clCreateKernel(g_cl.program, "tiled_gemm", &err);
  g_cl.k_tiled_gemm_transB =
      clCreateKernel(g_cl.program, "tiled_gemm_transB", &err);
  g_cl.k_softmax_stable = clCreateKernel(g_cl.program, "softmax_stable", &err);
  g_cl.k_rmsnorm_forward =
      clCreateKernel(g_cl.program, "rmsnorm_forward", &err);
  g_cl.k_rmsnorm_backward =
      clCreateKernel(g_cl.program, "rmsnorm_backward", &err);
  g_cl.k_rope_forward = clCreateKernel(g_cl.program, "rope_forward", &err);
  g_cl.k_swiglu_forward = clCreateKernel(g_cl.program, "swiglu_forward", &err);
  g_cl.k_swiglu_backward =
      clCreateKernel(g_cl.program, "swiglu_backward", &err);
  g_cl.k_topk_routing = clCreateKernel(g_cl.program, "topk_routing", &err);
  g_cl.k_head_attention = clCreateKernel(g_cl.program, "head_attention", &err);
  g_cl.k_elementwise_add =
      clCreateKernel(g_cl.program, "elementwise_add", &err);
  g_cl.k_elementwise_mul =
      clCreateKernel(g_cl.program, "elementwise_mul", &err);
  g_cl.k_elementwise_scale =
      clCreateKernel(g_cl.program, "elementwise_scale", &err);
  g_cl.k_elementwise_add_scaled =
      clCreateKernel(g_cl.program, "elementwise_add_scaled", &err);
  g_cl.k_bias_add = clCreateKernel(g_cl.program, "bias_add", &err);
  g_cl.k_cross_entropy_forward =
      clCreateKernel(g_cl.program, "cross_entropy_forward", &err);
  g_cl.k_cross_entropy_backward =
      clCreateKernel(g_cl.program, "cross_entropy_backward", &err);
  g_cl.k_adam_update = clCreateKernel(g_cl.program, "adam_update_kernel", &err);

  g_cl.initialized = 1;
  return 1;
}

void cl_cleanup(void) {
  if (!g_cl.initialized)
    return;

  clReleaseKernel(g_cl.k_tiled_gemm);
  clReleaseKernel(g_cl.k_tiled_gemm_transB);
  clReleaseKernel(g_cl.k_softmax_stable);
  clReleaseKernel(g_cl.k_rmsnorm_forward);
  clReleaseKernel(g_cl.k_rmsnorm_backward);
  clReleaseKernel(g_cl.k_rope_forward);
  clReleaseKernel(g_cl.k_swiglu_forward);
  clReleaseKernel(g_cl.k_swiglu_backward);
  clReleaseKernel(g_cl.k_topk_routing);
  clReleaseKernel(g_cl.k_head_attention);
  clReleaseKernel(g_cl.k_elementwise_add);
  clReleaseKernel(g_cl.k_elementwise_mul);
  clReleaseKernel(g_cl.k_elementwise_scale);
  clReleaseKernel(g_cl.k_elementwise_add_scaled);
  clReleaseKernel(g_cl.k_bias_add);
  clReleaseKernel(g_cl.k_cross_entropy_forward);
  clReleaseKernel(g_cl.k_cross_entropy_backward);
  clReleaseKernel(g_cl.k_adam_update);

  clReleaseProgram(g_cl.program);
  clReleaseCommandQueue(g_cl.queue);
  clReleaseContext(g_cl.context);

  g_cl.initialized = 0;
}

int cl_is_available(void) { return g_cl.initialized; }

cl_mem cl_alloc(size_t size) {
  cl_int err;
  cl_mem buf =
      clCreateBuffer(g_cl.context, CL_MEM_READ_WRITE, size, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to allocate %zu bytes on device\n", size);
    return NULL;
  }
  return buf;
}

cl_mem cl_alloc_copy(const float *data, size_t size) {
  cl_int err;
  cl_mem buf =
      clCreateBuffer(g_cl.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     size, (void *)data, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to allocate and copy %zu bytes\n", size);
    return NULL;
  }
  return buf;
}

void cl_free(cl_mem buf) {
  if (buf)
    clReleaseMemObject(buf);
}

void cl_copy_to_device(cl_mem buf, const float *data, size_t size) {
  CL_CHECK(clEnqueueWriteBuffer(g_cl.queue, buf, CL_TRUE, 0, size, data, 0,
                                NULL, NULL));
}

void cl_copy_to_host(float *data, cl_mem buf, size_t size) {
  CL_CHECK(clEnqueueReadBuffer(g_cl.queue, buf, CL_TRUE, 0, size, data, 0, NULL,
                               NULL));
}

void cl_zero(cl_mem buf, size_t size) {
  float zero = 0.0f;
  CL_CHECK(clEnqueueFillBuffer(g_cl.queue, buf, &zero, sizeof(float), 0, size,
                               0, NULL, NULL));
}

void cl_finish(void) { clFinish(g_cl.queue); }

// round up to next multiple
static size_t round_up(size_t x, size_t m) { return ((x + m - 1) / m) * m; }

void cl_gemm(cl_mem A, cl_mem B, cl_mem C, int M, int K, int N) {
  CL_CHECK(clSetKernelArg(g_cl.k_tiled_gemm, 0, sizeof(cl_mem), &A));
  CL_CHECK(clSetKernelArg(g_cl.k_tiled_gemm, 1, sizeof(cl_mem), &B));
  CL_CHECK(clSetKernelArg(g_cl.k_tiled_gemm, 2, sizeof(cl_mem), &C));
  CL_CHECK(clSetKernelArg(g_cl.k_tiled_gemm, 3, sizeof(int), &M));
  CL_CHECK(clSetKernelArg(g_cl.k_tiled_gemm, 4, sizeof(int), &K));
  CL_CHECK(clSetKernelArg(g_cl.k_tiled_gemm, 5, sizeof(int), &N));

  size_t local[2] = {16, 16};
  size_t global[2] = {round_up((size_t)M, 16), round_up((size_t)N, 16)};
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_tiled_gemm, 2, NULL,
                                  global, local, 0, NULL, NULL));
}

void cl_gemm_transB(cl_mem A, cl_mem B, cl_mem C, int M, int K, int N) {
  CL_CHECK(clSetKernelArg(g_cl.k_tiled_gemm_transB, 0, sizeof(cl_mem), &A));
  CL_CHECK(clSetKernelArg(g_cl.k_tiled_gemm_transB, 1, sizeof(cl_mem), &B));
  CL_CHECK(clSetKernelArg(g_cl.k_tiled_gemm_transB, 2, sizeof(cl_mem), &C));
  CL_CHECK(clSetKernelArg(g_cl.k_tiled_gemm_transB, 3, sizeof(int), &M));
  CL_CHECK(clSetKernelArg(g_cl.k_tiled_gemm_transB, 4, sizeof(int), &K));
  CL_CHECK(clSetKernelArg(g_cl.k_tiled_gemm_transB, 5, sizeof(int), &N));

  size_t local[2] = {16, 16};
  size_t global[2] = {round_up((size_t)M, 16), round_up((size_t)N, 16)};
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_tiled_gemm_transB, 2, NULL,
                                  global, local, 0, NULL, NULL));
}

void cl_softmax(cl_mem x, int rows, int cols) {
  CL_CHECK(clSetKernelArg(g_cl.k_softmax_stable, 0, sizeof(cl_mem), &x));
  CL_CHECK(clSetKernelArg(g_cl.k_softmax_stable, 1, sizeof(int), &rows));
  CL_CHECK(clSetKernelArg(g_cl.k_softmax_stable, 2, sizeof(int), &cols));

  size_t global = (size_t)rows;
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_softmax_stable, 1, NULL,
                                  &global, NULL, 0, NULL, NULL));
}

void cl_rmsnorm_forward(cl_mem x, cl_mem gamma, cl_mem y, int rows, int dim,
                        float eps) {
  CL_CHECK(clSetKernelArg(g_cl.k_rmsnorm_forward, 0, sizeof(cl_mem), &x));
  CL_CHECK(clSetKernelArg(g_cl.k_rmsnorm_forward, 1, sizeof(cl_mem), &gamma));
  CL_CHECK(clSetKernelArg(g_cl.k_rmsnorm_forward, 2, sizeof(cl_mem), &y));
  CL_CHECK(clSetKernelArg(g_cl.k_rmsnorm_forward, 3, sizeof(int), &rows));
  CL_CHECK(clSetKernelArg(g_cl.k_rmsnorm_forward, 4, sizeof(int), &dim));
  CL_CHECK(clSetKernelArg(g_cl.k_rmsnorm_forward, 5, sizeof(float), &eps));

  size_t global = (size_t)rows;
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_rmsnorm_forward, 1, NULL,
                                  &global, NULL, 0, NULL, NULL));
}

void cl_rope_forward(cl_mem x, int seq_len, int dim, float base) {
  CL_CHECK(clSetKernelArg(g_cl.k_rope_forward, 0, sizeof(cl_mem), &x));
  CL_CHECK(clSetKernelArg(g_cl.k_rope_forward, 1, sizeof(int), &seq_len));
  CL_CHECK(clSetKernelArg(g_cl.k_rope_forward, 2, sizeof(int), &dim));
  CL_CHECK(clSetKernelArg(g_cl.k_rope_forward, 3, sizeof(float), &base));

  size_t global[2] = {(size_t)seq_len, (size_t)(dim / 2)};
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_rope_forward, 2, NULL,
                                  global, NULL, 0, NULL, NULL));
}

void cl_swiglu_forward(cl_mem gate, cl_mem value, cl_mem out, int size) {
  CL_CHECK(clSetKernelArg(g_cl.k_swiglu_forward, 0, sizeof(cl_mem), &gate));
  CL_CHECK(clSetKernelArg(g_cl.k_swiglu_forward, 1, sizeof(cl_mem), &value));
  CL_CHECK(clSetKernelArg(g_cl.k_swiglu_forward, 2, sizeof(cl_mem), &out));
  CL_CHECK(clSetKernelArg(g_cl.k_swiglu_forward, 3, sizeof(int), &size));

  size_t global = round_up((size_t)size, 256);
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_swiglu_forward, 1, NULL,
                                  &global, NULL, 0, NULL, NULL));
}

void cl_swiglu_backward(cl_mem dout, cl_mem gate, cl_mem value, cl_mem dgate,
                        cl_mem dvalue, int size) {
  CL_CHECK(clSetKernelArg(g_cl.k_swiglu_backward, 0, sizeof(cl_mem), &dout));
  CL_CHECK(clSetKernelArg(g_cl.k_swiglu_backward, 1, sizeof(cl_mem), &gate));
  CL_CHECK(clSetKernelArg(g_cl.k_swiglu_backward, 2, sizeof(cl_mem), &value));
  CL_CHECK(clSetKernelArg(g_cl.k_swiglu_backward, 3, sizeof(cl_mem), &dgate));
  CL_CHECK(clSetKernelArg(g_cl.k_swiglu_backward, 4, sizeof(cl_mem), &dvalue));
  CL_CHECK(clSetKernelArg(g_cl.k_swiglu_backward, 5, sizeof(int), &size));

  size_t global = round_up((size_t)size, 256);
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_swiglu_backward, 1, NULL,
                                  &global, NULL, 0, NULL, NULL));
}

void cl_topk_routing(cl_mem logits, cl_mem indices, cl_mem weights, int seq_len,
                     int n_experts, int top_k) {
  CL_CHECK(clSetKernelArg(g_cl.k_topk_routing, 0, sizeof(cl_mem), &logits));
  CL_CHECK(clSetKernelArg(g_cl.k_topk_routing, 1, sizeof(cl_mem), &indices));
  CL_CHECK(clSetKernelArg(g_cl.k_topk_routing, 2, sizeof(cl_mem), &weights));
  CL_CHECK(clSetKernelArg(g_cl.k_topk_routing, 3, sizeof(int), &seq_len));
  CL_CHECK(clSetKernelArg(g_cl.k_topk_routing, 4, sizeof(int), &n_experts));
  CL_CHECK(clSetKernelArg(g_cl.k_topk_routing, 5, sizeof(int), &top_k));

  size_t global = (size_t)seq_len;
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_topk_routing, 1, NULL,
                                  &global, NULL, 0, NULL, NULL));
}

void cl_head_attention(cl_mem Q, cl_mem K, cl_mem V, cl_mem out, cl_mem mask,
                       int seq_q, int seq_k, int dk, int dv, float scale) {
  CL_CHECK(clSetKernelArg(g_cl.k_head_attention, 0, sizeof(cl_mem), &Q));
  CL_CHECK(clSetKernelArg(g_cl.k_head_attention, 1, sizeof(cl_mem), &K));
  CL_CHECK(clSetKernelArg(g_cl.k_head_attention, 2, sizeof(cl_mem), &V));
  CL_CHECK(clSetKernelArg(g_cl.k_head_attention, 3, sizeof(cl_mem), &out));
  CL_CHECK(clSetKernelArg(g_cl.k_head_attention, 4, sizeof(cl_mem), &mask));
  CL_CHECK(clSetKernelArg(g_cl.k_head_attention, 5, sizeof(int), &seq_q));
  CL_CHECK(clSetKernelArg(g_cl.k_head_attention, 6, sizeof(int), &seq_k));
  CL_CHECK(clSetKernelArg(g_cl.k_head_attention, 7, sizeof(int), &dk));
  CL_CHECK(clSetKernelArg(g_cl.k_head_attention, 8, sizeof(int), &dv));
  CL_CHECK(clSetKernelArg(g_cl.k_head_attention, 9, sizeof(float), &scale));

  size_t global = (size_t)seq_q;
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_head_attention, 1, NULL,
                                  &global, NULL, 0, NULL, NULL));
}

void cl_elementwise_add(cl_mem a, cl_mem b, cl_mem c, int size) {
  CL_CHECK(clSetKernelArg(g_cl.k_elementwise_add, 0, sizeof(cl_mem), &a));
  CL_CHECK(clSetKernelArg(g_cl.k_elementwise_add, 1, sizeof(cl_mem), &b));
  CL_CHECK(clSetKernelArg(g_cl.k_elementwise_add, 2, sizeof(cl_mem), &c));
  CL_CHECK(clSetKernelArg(g_cl.k_elementwise_add, 3, sizeof(int), &size));

  size_t global = round_up((size_t)size, 256);
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_elementwise_add, 1, NULL,
                                  &global, NULL, 0, NULL, NULL));
}

void cl_elementwise_scale(cl_mem x, float scale, int size) {
  CL_CHECK(clSetKernelArg(g_cl.k_elementwise_scale, 0, sizeof(cl_mem), &x));
  CL_CHECK(clSetKernelArg(g_cl.k_elementwise_scale, 1, sizeof(float), &scale));
  CL_CHECK(clSetKernelArg(g_cl.k_elementwise_scale, 2, sizeof(int), &size));

  size_t global = round_up((size_t)size, 256);
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_elementwise_scale, 1, NULL,
                                  &global, NULL, 0, NULL, NULL));
}

void cl_bias_add(cl_mem y, cl_mem bias, int rows, int cols) {
  CL_CHECK(clSetKernelArg(g_cl.k_bias_add, 0, sizeof(cl_mem), &y));
  CL_CHECK(clSetKernelArg(g_cl.k_bias_add, 1, sizeof(cl_mem), &bias));
  CL_CHECK(clSetKernelArg(g_cl.k_bias_add, 2, sizeof(int), &rows));
  CL_CHECK(clSetKernelArg(g_cl.k_bias_add, 3, sizeof(int), &cols));

  size_t global[2] = {(size_t)rows, (size_t)cols};
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_bias_add, 2, NULL, global,
                                  NULL, 0, NULL, NULL));
}

void cl_cross_entropy_forward(cl_mem logits, cl_mem targets, cl_mem losses,
                              int seq_len, int vocab_size) {
  CL_CHECK(
      clSetKernelArg(g_cl.k_cross_entropy_forward, 0, sizeof(cl_mem), &logits));
  CL_CHECK(clSetKernelArg(g_cl.k_cross_entropy_forward, 1, sizeof(cl_mem),
                          &targets));
  CL_CHECK(
      clSetKernelArg(g_cl.k_cross_entropy_forward, 2, sizeof(cl_mem), &losses));
  CL_CHECK(
      clSetKernelArg(g_cl.k_cross_entropy_forward, 3, sizeof(int), &seq_len));
  CL_CHECK(clSetKernelArg(g_cl.k_cross_entropy_forward, 4, sizeof(int),
                          &vocab_size));

  size_t global = (size_t)seq_len;
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_cross_entropy_forward, 1,
                                  NULL, &global, NULL, 0, NULL, NULL));
}

void cl_cross_entropy_backward(cl_mem logits, cl_mem targets, cl_mem dlogits,
                               int seq_len, int vocab_size) {
  CL_CHECK(clSetKernelArg(g_cl.k_cross_entropy_backward, 0, sizeof(cl_mem),
                          &logits));
  CL_CHECK(clSetKernelArg(g_cl.k_cross_entropy_backward, 1, sizeof(cl_mem),
                          &targets));
  CL_CHECK(clSetKernelArg(g_cl.k_cross_entropy_backward, 2, sizeof(cl_mem),
                          &dlogits));
  CL_CHECK(
      clSetKernelArg(g_cl.k_cross_entropy_backward, 3, sizeof(int), &seq_len));
  CL_CHECK(clSetKernelArg(g_cl.k_cross_entropy_backward, 4, sizeof(int),
                          &vocab_size));

  size_t global = (size_t)seq_len;
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_cross_entropy_backward, 1,
                                  NULL, &global, NULL, 0, NULL, NULL));
}

void cl_adam_update(cl_mem params, cl_mem grads, cl_mem m, cl_mem v, float lr,
                    float beta1, float beta2, float eps, float weight_decay,
                    float bc1, float bc2, int size) {
  CL_CHECK(clSetKernelArg(g_cl.k_adam_update, 0, sizeof(cl_mem), &params));
  CL_CHECK(clSetKernelArg(g_cl.k_adam_update, 1, sizeof(cl_mem), &grads));
  CL_CHECK(clSetKernelArg(g_cl.k_adam_update, 2, sizeof(cl_mem), &m));
  CL_CHECK(clSetKernelArg(g_cl.k_adam_update, 3, sizeof(cl_mem), &v));
  CL_CHECK(clSetKernelArg(g_cl.k_adam_update, 4, sizeof(float), &lr));
  CL_CHECK(clSetKernelArg(g_cl.k_adam_update, 5, sizeof(float), &beta1));
  CL_CHECK(clSetKernelArg(g_cl.k_adam_update, 6, sizeof(float), &beta2));
  CL_CHECK(clSetKernelArg(g_cl.k_adam_update, 7, sizeof(float), &eps));
  CL_CHECK(clSetKernelArg(g_cl.k_adam_update, 8, sizeof(float), &weight_decay));
  CL_CHECK(clSetKernelArg(g_cl.k_adam_update, 9, sizeof(float), &bc1));
  CL_CHECK(clSetKernelArg(g_cl.k_adam_update, 10, sizeof(float), &bc2));
  CL_CHECK(clSetKernelArg(g_cl.k_adam_update, 11, sizeof(int), &size));

  size_t global = round_up((size_t)size, 256);
  CL_CHECK(clEnqueueNDRangeKernel(g_cl.queue, g_cl.k_adam_update, 1, NULL,
                                  &global, NULL, 0, NULL, NULL));
}
