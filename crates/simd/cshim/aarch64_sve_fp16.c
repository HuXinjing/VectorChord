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
#if defined(__AARCH64EL__)
#include <arm_sve.h>
#endif
#include <stddef.h>
#include <stdint.h>

typedef __fp16 f16;
typedef float f32;

__attribute__((target("+sve"))) float
fp16_reduce_sum_of_xy_a3_512(size_t n, f16 *restrict a, f16 *restrict b) {
  svfloat16_t sum = svdup_f16(0.0);
  for (size_t i = 0; i < n; i += svcnth()) {
    svbool_t mask = svwhilelt_b16((int64_t)i, (int64_t)n);
    svfloat16_t x = svld1_f16(mask, a + i);
    svfloat16_t y = svld1_f16(mask, b + i);
    sum = svmla_f16_m(mask, sum, x, y);
  }
  return svaddv_f16(svptrue_b16(), sum);
}

__attribute__((target("+sve"))) float
fp16_reduce_sum_of_d2_a3_512(size_t n, f16 *restrict a, f16 *restrict b) {
  svfloat16_t sum = svdup_f16(0.0);
  for (size_t i = 0; i < n; i += svcnth()) {
    svbool_t mask = svwhilelt_b16((int64_t)i, (int64_t)n);
    svfloat16_t x = svld1_f16(mask, a + i);
    svfloat16_t y = svld1_f16(mask, b + i);
    svfloat16_t d = svsub_f16_z(mask, x, y);
    sum = svmla_f16_m(mask, sum, d, d);
  }
  return svaddv_f16(svptrue_b16(), sum);
}
