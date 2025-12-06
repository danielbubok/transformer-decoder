#ifndef PRECISION_H
#define PRECISION_H

#include <math.h>
#include <stdint.h>
#include <string.h>

// data types for mixed precision training
typedef enum {
  DTYPE_FP32 = 0,
  DTYPE_FP16 = 1,
  DTYPE_BF16 = 2,
  DTYPE_FP8_E4M3 = 3, // 4 exponent, 3 mantissa - weights
  DTYPE_FP8_E5M2 = 4  // 5 exponent, 2 mantissa - gradients
} DType;

// fp16 type: use native if available, otherwise uint16
#if defined(__FLT16_MAX__)
typedef _Float16 fp16_t;
#define FP16_NATIVE 1
#else
typedef uint16_t fp16_t;
#define FP16_NATIVE 0
#endif

// bf16 type
typedef uint16_t bf16_t;

// fp8 types
typedef uint8_t fp8_e4m3_t;
typedef uint8_t fp8_e5m2_t;

// fp16 constants
#define FP16_EXP_BIAS 15
#define FP16_EXP_BITS 5
#define FP16_MANT_BITS 10
#define FP16_MAX 65504.0f
#define FP16_MIN 6.103515625e-5f

// bf16 constants
#define BF16_EXP_BIAS 127
#define BF16_EXP_BITS 8
#define BF16_MANT_BITS 7

// fp8 e4m3 constants (range: -448 to 448)
#define FP8_E4M3_EXP_BIAS 7
#define FP8_E4M3_MAX 448.0f

// fp8 e5m2 constants (range: -57344 to 57344)
#define FP8_E5M2_EXP_BIAS 15
#define FP8_E5M2_MAX 57344.0f

// === FP16 Conversions ===

static inline fp16_t fp32_to_fp16(float f) {
#if FP16_NATIVE
  return (fp16_t)f;
#else
  uint32_t x;
  memcpy(&x, &f, sizeof(float));

  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = (x >> 13) & 0x3ff;

  if (exp <= 0) {
    // denormal or zero
    if (exp < -10)
      return (fp16_t)sign;
    mant = (mant | 0x400) >> (1 - exp);
    return (fp16_t)(sign | mant);
  } else if (exp >= 31) {
    // inf or nan
    if (exp == 255 - 127 + 15 && mant) {
      return (fp16_t)(sign | 0x7e00); // nan
    }
    return (fp16_t)(sign | 0x7c00); // inf
  }

  return (fp16_t)(sign | ((uint32_t)exp << 10) | mant);
#endif
}

static inline float fp16_to_fp32(fp16_t h) {
#if FP16_NATIVE
  return (float)h;
#else
  uint32_t sign = ((uint32_t)h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;

  if (exp == 0) {
    if (mant == 0) {
      uint32_t result = sign;
      float f;
      memcpy(&f, &result, sizeof(float));
      return f;
    }
    // denormal
    while ((mant & 0x400) == 0) {
      mant <<= 1;
      exp--;
    }
    exp++;
    mant &= 0x3ff;
  } else if (exp == 31) {
    uint32_t result = sign | 0x7f800000 | (mant << 13);
    float f;
    memcpy(&f, &result, sizeof(float));
    return f;
  }

  exp = exp + 127 - 15;
  uint32_t result = sign | (exp << 23) | (mant << 13);
  float f;
  memcpy(&f, &result, sizeof(float));
  return f;
#endif
}

// === BF16 Conversions ===

static inline bf16_t fp32_to_bf16(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof(float));
  // round to nearest even
  uint32_t rounding = 0x7fff + ((x >> 16) & 1);
  x += rounding;
  return (bf16_t)(x >> 16);
}

static inline float bf16_to_fp32(bf16_t b) {
  uint32_t x = (uint32_t)b << 16;
  float f;
  memcpy(&f, &x, sizeof(float));
  return f;
}

// === FP8 E4M3 Conversions ===

static inline fp8_e4m3_t fp32_to_fp8_e4m3(float f) {
  if (f != f)
    return 0x7f; // nan

  uint32_t x;
  memcpy(&x, &f, sizeof(float));
  uint8_t sign = (x >> 24) & 0x80;

  float abs_f = fabsf(f);
  if (abs_f > FP8_E4M3_MAX)
    abs_f = FP8_E4M3_MAX;
  if (abs_f < 1.0f / 64.0f)
    return sign; // zero or denormal

  int exp = (int)floorf(log2f(abs_f));
  exp = exp + FP8_E4M3_EXP_BIAS;
  if (exp < 1)
    exp = 1;
  if (exp > 15)
    exp = 15;

  float scale = powf(2.0f, (float)(exp - FP8_E4M3_EXP_BIAS));
  int mant = (int)roundf((abs_f / scale - 1.0f) * 8.0f);
  if (mant > 7)
    mant = 7;
  if (mant < 0)
    mant = 0;

  return sign | (uint8_t)((exp << 3) | mant);
}

static inline float fp8_e4m3_to_fp32(fp8_e4m3_t x) {
  if (x == 0x7f || x == 0xff)
    return NAN;

  uint8_t sign = x & 0x80;
  uint8_t exp = (x >> 3) & 0xf;
  uint8_t mant = x & 0x7;

  if (exp == 0 && mant == 0)
    return sign ? -0.0f : 0.0f;

  float scale = powf(2.0f, (float)(exp - FP8_E4M3_EXP_BIAS));
  float val = (1.0f + (float)mant / 8.0f) * scale;

  return sign ? -val : val;
}

// === FP8 E5M2 Conversions ===

static inline fp8_e5m2_t fp32_to_fp8_e5m2(float f) {
  if (f != f)
    return 0x7f; // nan

  uint32_t x;
  memcpy(&x, &f, sizeof(float));
  uint8_t sign = (x >> 24) & 0x80;

  float abs_f = fabsf(f);
  if (abs_f > FP8_E5M2_MAX)
    abs_f = FP8_E5M2_MAX;
  if (abs_f < 1.0f / 16384.0f)
    return sign; // zero or denormal

  int exp = (int)floorf(log2f(abs_f));
  exp = exp + FP8_E5M2_EXP_BIAS;
  if (exp < 1)
    exp = 1;
  if (exp > 30)
    exp = 30;

  float scale = powf(2.0f, (float)(exp - FP8_E5M2_EXP_BIAS));
  int mant = (int)roundf((abs_f / scale - 1.0f) * 4.0f);
  if (mant > 3)
    mant = 3;
  if (mant < 0)
    mant = 0;

  return sign | (uint8_t)((exp << 2) | mant);
}

static inline float fp8_e5m2_to_fp32(fp8_e5m2_t x) {
  uint8_t sign = x & 0x80;
  uint8_t exp = (x >> 2) & 0x1f;
  uint8_t mant = x & 0x3;

  if (exp == 31)
    return sign ? -INFINITY : INFINITY;
  if (exp == 0 && mant == 0)
    return sign ? -0.0f : 0.0f;

  float scale = powf(2.0f, (float)(exp - FP8_E5M2_EXP_BIAS));
  float val = (1.0f + (float)mant / 4.0f) * scale;

  return sign ? -val : val;
}

// === Utility Functions ===

// get size in bytes for dtype
static inline int dtype_size(DType dtype) {
  switch (dtype) {
  case DTYPE_FP32:
    return 4;
  case DTYPE_FP16:
    return 2;
  case DTYPE_BF16:
    return 2;
  case DTYPE_FP8_E4M3:
    return 1;
  case DTYPE_FP8_E5M2:
    return 1;
  default:
    return 4;
  }
}

// get string name for dtype
static inline const char *dtype_name(DType dtype) {
  switch (dtype) {
  case DTYPE_FP32:
    return "fp32";
  case DTYPE_FP16:
    return "fp16";
  case DTYPE_BF16:
    return "bf16";
  case DTYPE_FP8_E4M3:
    return "fp8_e4m3";
  case DTYPE_FP8_E5M2:
    return "fp8_e5m2";
  default:
    return "unknown";
  }
}

// parse dtype from string
static inline DType dtype_from_string(const char *s) {
  if (!s)
    return DTYPE_FP32;
  if (strcmp(s, "fp32") == 0)
    return DTYPE_FP32;
  if (strcmp(s, "fp16") == 0)
    return DTYPE_FP16;
  if (strcmp(s, "bf16") == 0)
    return DTYPE_BF16;
  if (strcmp(s, "fp8") == 0)
    return DTYPE_FP8_E4M3;
  if (strcmp(s, "fp8_e4m3") == 0)
    return DTYPE_FP8_E4M3;
  if (strcmp(s, "fp8_e5m2") == 0)
    return DTYPE_FP8_E5M2;
  return DTYPE_FP32;
}

// quantization context for fp8
typedef struct {
  float scale;
  float inv_scale;
  DType dtype;
} QuantContext;

// compute quantization scale for tensor
float compute_quant_scale(const float *data, int size, DType dtype);

// bulk conversions
void convert_fp32_to_fp16(fp16_t *dst, const float *src, int n);
void convert_fp16_to_fp32(float *dst, const fp16_t *src, int n);
void convert_fp32_to_bf16(bf16_t *dst, const float *src, int n);
void convert_bf16_to_fp32(float *dst, const bf16_t *src, int n);
void convert_fp32_to_fp8_e4m3(fp8_e4m3_t *dst, const float *src, int n,
                              float scale);
void convert_fp8_e4m3_to_fp32(float *dst, const fp8_e4m3_t *src, int n,
                              float scale);

#endif
