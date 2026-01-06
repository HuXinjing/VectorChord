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

#![allow(unsafe_code)]

use criterion::{Criterion, criterion_group, criterion_main};

fn byte_reduce_sum_of_xy(c: &mut Criterion) {
    use rand::Rng;
    let mut rng = rand::rng();
    let x = (0..4095).map(|_| rng.random()).collect::<Vec<_>>();
    let y = (0..4095).map(|_| rng.random()).collect::<Vec<_>>();
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") && simd::is_feature_detected!("avx512vnni") {
        c.bench_function("byte::reduce_sum_of_xy::v4_avx512vnni", |b| {
            b.iter(|| unsafe {
                simd::byte::reduce_sum_of_xy::reduce_sum_of_xy_v4_avx512vnni(&x, &y)
            })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") {
        c.bench_function("byte::reduce_sum_of_xy::v4", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_xy::reduce_sum_of_xy_v4(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v3") {
        c.bench_function("byte::reduce_sum_of_xy::v3", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_xy::reduce_sum_of_xy_v3(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v2") {
        c.bench_function("byte::reduce_sum_of_xy::v2", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_xy::reduce_sum_of_xy_v2(&x, &y) })
        });
    }
}

fn byte_reduce_sum_of_x(c: &mut Criterion) {
    use rand::Rng;
    let mut rng = rand::rng();
    let this = (0..4095).map(|_| rng.random()).collect::<Vec<_>>();
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") {
        c.bench_function("byte::reduce_sum_of_x::v4", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_x::reduce_sum_of_x_v4(&this) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v3") {
        c.bench_function("byte::reduce_sum_of_x::v3", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_x::reduce_sum_of_x_v3(&this) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v2") {
        c.bench_function("byte::reduce_sum_of_x::v2", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_x::reduce_sum_of_x_v2(&this) })
        });
    }
}

fn halfbyte_reduce_sum_of_xy(c: &mut Criterion) {
    use rand::Rng;
    let mut rng = rand::rng();
    let x = (0..2047).map(|_| rng.random()).collect::<Vec<_>>();
    let y = (0..2047).map(|_| rng.random()).collect::<Vec<_>>();
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") && simd::is_feature_detected!("avx512vnni") {
        c.bench_function("halfbyte::reduce_sum_of_xy::v4_avx512vnni", |b| {
            b.iter(|| unsafe {
                simd::halfbyte::reduce_sum_of_xy::reduce_sum_of_xy_v4_avx512vnni(&x, &y)
            })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") {
        c.bench_function("halfbyte::reduce_sum_of_xy::v4", |b| {
            b.iter(|| unsafe { simd::halfbyte::reduce_sum_of_xy::reduce_sum_of_xy_v4(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v3") {
        c.bench_function("halfbyte::reduce_sum_of_xy::v3", |b| {
            b.iter(|| unsafe { simd::halfbyte::reduce_sum_of_xy::reduce_sum_of_xy_v3(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v2") {
        c.bench_function("halfbyte::reduce_sum_of_xy::v2", |b| {
            b.iter(|| unsafe { simd::halfbyte::reduce_sum_of_xy::reduce_sum_of_xy_v2(&x, &y) })
        });
    }
}

criterion_group!(
    benches,
    byte_reduce_sum_of_xy,
    byte_reduce_sum_of_x,
    halfbyte_reduce_sum_of_xy
);
criterion_main!(benches);
