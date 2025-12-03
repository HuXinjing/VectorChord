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

use crate::datatype::memory_halfvec::HalfvecInput;
use crate::datatype::memory_rabitq8::Rabitq8Output;
use crate::datatype::memory_vector::VectorInput;
use simd::{Floating, f16};
use vector::rabitq8::Rabitq8Borrowed;

#[pgrx::pg_extern(sql = "")]
fn _vchord_vector_quantize_to_rabitq8(vector: VectorInput) -> Rabitq8Output {
    let mut vector = vector.as_borrowed().slice().to_vec();
    rabitq::rotate::rotate_inplace(&mut vector);
    let code = rabitq::byte::ugly_code(&vector);
    Rabitq8Output::new(Rabitq8Borrowed::new(
        code.0.dis_u_2,
        code.0.norm_of_lattice,
        code.0.sum_of_code,
        f32::reduce_sum_of_abs_x(&vector),
        &code.1,
    ))
}

#[pgrx::pg_extern(sql = "")]
fn _vchord_halfvec_quantize_to_rabitq8(vector: HalfvecInput) -> Rabitq8Output {
    let mut vector = f16::vector_to_f32(vector.as_borrowed().slice());
    rabitq::rotate::rotate_inplace(&mut vector);
    let code = rabitq::byte::ugly_code(&vector);
    Rabitq8Output::new(Rabitq8Borrowed::new(
        code.0.dis_u_2,
        code.0.norm_of_lattice,
        code.0.sum_of_code,
        f32::reduce_sum_of_abs_x(&vector),
        &code.1,
    ))
}
