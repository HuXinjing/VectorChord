// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

#if defined(__clang__)
#if !(__clang_major__ >= 16)
#error "Clang version must be at least 16."
#endif
#elif defined(__GNUC__)
#if !(__GNUC__ >= 14)
#error "GCC version must be at least 14."
#endif
#else
#error "This file requires Clang or GCC."
#endif

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

typedef __fp16 f16;
typedef float f32;

__attribute__((target("+fp16"))) float
fp16_reduce_sum_of_xy_a2_fp16(size_t n, f16 *restrict a, f16 *restrict b) {
  float16x8_t sum_0 = vdupq_n_f16(0.0);
  float16x8_t sum_1 = vdupq_n_f16(0.0);
  float16x8_t sum_2 = vdupq_n_f16(0.0);
  float16x8_t sum_3 = vdupq_n_f16(0.0);
  while (n >= 32) {
    float16x8_t x_0 = vld1q_f16(a + 0);
    float16x8_t x_1 = vld1q_f16(a + 8);
    float16x8_t x_2 = vld1q_f16(a + 16);
    float16x8_t x_3 = vld1q_f16(a + 24);
    float16x8_t y_0 = vld1q_f16(b + 0);
    float16x8_t y_1 = vld1q_f16(b + 8);
    float16x8_t y_2 = vld1q_f16(b + 16);
    float16x8_t y_3 = vld1q_f16(b + 24);
    sum_0 = vfmaq_f16(sum_0, x_0, y_0);
    sum_1 = vfmaq_f16(sum_1, x_1, y_1);
    sum_2 = vfmaq_f16(sum_2, x_2, y_2);
    sum_3 = vfmaq_f16(sum_3, x_3, y_3);
    n -= 32, a += 32, b += 32;
  }
  if (n > 0) {
    f16 _a[32] = {}, _b[32] = {};
    for (size_t i = 0; i < n; i += 1) {
      _a[i] = a[i], _b[i] = b[i];
    }
    a = _a, b = _b;
    float16x8_t x_0 = vld1q_f16(_a + 0);
    float16x8_t x_1 = vld1q_f16(_a + 8);
    float16x8_t x_2 = vld1q_f16(_a + 16);
    float16x8_t x_3 = vld1q_f16(_a + 24);
    float16x8_t y_0 = vld1q_f16(_b + 0);
    float16x8_t y_1 = vld1q_f16(_b + 8);
    float16x8_t y_2 = vld1q_f16(_b + 16);
    float16x8_t y_3 = vld1q_f16(_b + 24);
    sum_0 = vfmaq_f16(sum_0, x_0, y_0);
    sum_1 = vfmaq_f16(sum_1, x_1, y_1);
    sum_2 = vfmaq_f16(sum_2, x_2, y_2);
    sum_3 = vfmaq_f16(sum_3, x_3, y_3);
  }
  float32x4_t s_0 = vcvt_f32_f16(vget_low_f16(sum_0));
  float32x4_t s_1 = vcvt_f32_f16(vget_high_f16(sum_0));
  float32x4_t s_2 = vcvt_f32_f16(vget_low_f16(sum_1));
  float32x4_t s_3 = vcvt_f32_f16(vget_high_f16(sum_1));
  float32x4_t s_4 = vcvt_f32_f16(vget_low_f16(sum_2));
  float32x4_t s_5 = vcvt_f32_f16(vget_high_f16(sum_2));
  float32x4_t s_6 = vcvt_f32_f16(vget_low_f16(sum_3));
  float32x4_t s_7 = vcvt_f32_f16(vget_high_f16(sum_3));
  float32x4_t s =
      vpaddq_f32(vpaddq_f32(vpaddq_f32(s_0, s_1), vpaddq_f32(s_2, s_3)),
                 vpaddq_f32(vpaddq_f32(s_4, s_5), vpaddq_f32(s_6, s_7)));
  return vaddvq_f32(s);
}

__attribute__((target("+fp16"))) float
fp16_reduce_sum_of_d2_a2_fp16(size_t n, f16 *restrict a, f16 *restrict b) {
  float16x8_t sum_0 = vdupq_n_f16(0.0);
  float16x8_t sum_1 = vdupq_n_f16(0.0);
  float16x8_t sum_2 = vdupq_n_f16(0.0);
  float16x8_t sum_3 = vdupq_n_f16(0.0);
  while (n >= 32) {
    float16x8_t x_0 = vld1q_f16(a + 0);
    float16x8_t x_1 = vld1q_f16(a + 8);
    float16x8_t x_2 = vld1q_f16(a + 16);
    float16x8_t x_3 = vld1q_f16(a + 24);
    float16x8_t y_0 = vld1q_f16(b + 0);
    float16x8_t y_1 = vld1q_f16(b + 8);
    float16x8_t y_2 = vld1q_f16(b + 16);
    float16x8_t y_3 = vld1q_f16(b + 24);
    float16x8_t d_0 = vsubq_f16(x_0, y_0);
    float16x8_t d_1 = vsubq_f16(x_1, y_1);
    float16x8_t d_2 = vsubq_f16(x_2, y_2);
    float16x8_t d_3 = vsubq_f16(x_3, y_3);
    sum_0 = vfmaq_f16(sum_0, d_0, d_0);
    sum_1 = vfmaq_f16(sum_1, d_1, d_1);
    sum_2 = vfmaq_f16(sum_2, d_2, d_2);
    sum_3 = vfmaq_f16(sum_3, d_3, d_3);
    n -= 32, a += 32, b += 32;
  }
  if (n > 0) {
    f16 _a[32] = {}, _b[32] = {};
    for (size_t i = 0; i < n; i += 1) {
      _a[i] = a[i], _b[i] = b[i];
    }
    a = _a, b = _b;
    float16x8_t x_0 = vld1q_f16(a + 0);
    float16x8_t x_1 = vld1q_f16(a + 8);
    float16x8_t x_2 = vld1q_f16(a + 16);
    float16x8_t x_3 = vld1q_f16(a + 24);
    float16x8_t y_0 = vld1q_f16(b + 0);
    float16x8_t y_1 = vld1q_f16(b + 8);
    float16x8_t y_2 = vld1q_f16(b + 16);
    float16x8_t y_3 = vld1q_f16(b + 24);
    float16x8_t d_0 = vsubq_f16(x_0, y_0);
    float16x8_t d_1 = vsubq_f16(x_1, y_1);
    float16x8_t d_2 = vsubq_f16(x_2, y_2);
    float16x8_t d_3 = vsubq_f16(x_3, y_3);
    sum_0 = vfmaq_f16(sum_0, d_0, d_0);
    sum_1 = vfmaq_f16(sum_1, d_1, d_1);
    sum_2 = vfmaq_f16(sum_2, d_2, d_2);
    sum_3 = vfmaq_f16(sum_3, d_3, d_3);
  }
  float32x4_t s_0 = vcvt_f32_f16(vget_low_f16(sum_0));
  float32x4_t s_1 = vcvt_f32_f16(vget_high_f16(sum_0));
  float32x4_t s_2 = vcvt_f32_f16(vget_low_f16(sum_1));
  float32x4_t s_3 = vcvt_f32_f16(vget_high_f16(sum_1));
  float32x4_t s_4 = vcvt_f32_f16(vget_low_f16(sum_2));
  float32x4_t s_5 = vcvt_f32_f16(vget_high_f16(sum_2));
  float32x4_t s_6 = vcvt_f32_f16(vget_low_f16(sum_3));
  float32x4_t s_7 = vcvt_f32_f16(vget_high_f16(sum_3));
  float32x4_t s =
      vpaddq_f32(vpaddq_f32(vpaddq_f32(s_0, s_1), vpaddq_f32(s_2, s_3)),
                 vpaddq_f32(vpaddq_f32(s_4, s_5), vpaddq_f32(s_6, s_7)));
  return vaddvq_f32(s);
}
