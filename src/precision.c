#include "precision.h"
#include <stdlib.h>
#include <string.h>

// compute quantization scale for fp8
float compute_quant_scale(const float *data, int size, DType dtype) {
  float max_val = 0.0f;
  for (int i = 0; i < size; i++) {
    float abs_val = fabsf(data[i]);
    if (abs_val > max_val)
      max_val = abs_val;
  }

  float type_max;
  switch (dtype) {
  case DTYPE_FP8_E4M3:
    type_max = FP8_E4M3_MAX;
    break;
  case DTYPE_FP8_E5M2:
    type_max = FP8_E5M2_MAX;
    break;
  case DTYPE_FP16:
    type_max = FP16_MAX;
    break;
  default:
    return 1.0f;
  }

  if (max_val == 0.0f)
    return 1.0f;
  return type_max / max_val;
}

// bulk conversions
void convert_fp32_to_fp16(fp16_t *dst, const float *src, int n) {
  for (int i = 0; i < n; i++) {
    dst[i] = fp32_to_fp16(src[i]);
  }
}

void convert_fp16_to_fp32(float *dst, const fp16_t *src, int n) {
  for (int i = 0; i < n; i++) {
    dst[i] = fp16_to_fp32(src[i]);
  }
}

void convert_fp32_to_bf16(bf16_t *dst, const float *src, int n) {
  for (int i = 0; i < n; i++) {
    dst[i] = fp32_to_bf16(src[i]);
  }
}

void convert_bf16_to_fp32(float *dst, const bf16_t *src, int n) {
  for (int i = 0; i < n; i++) {
    dst[i] = bf16_to_fp32(src[i]);
  }
}

void convert_fp32_to_fp8_e4m3(fp8_e4m3_t *dst, const float *src, int n,
                              float scale) {
  for (int i = 0; i < n; i++) {
    dst[i] = fp32_to_fp8_e4m3(src[i] * scale);
  }
}

void convert_fp8_e4m3_to_fp32(float *dst, const fp8_e4m3_t *src, int n,
                              float scale) {
  float inv_scale = 1.0f / scale;
  for (int i = 0; i < n; i++) {
    dst[i] = fp8_e4m3_to_fp32(src[i]) * inv_scale;
  }
}
