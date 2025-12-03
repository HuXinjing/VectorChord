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

use simd::{Floating, f16};
use vector::rabitq8::Rabitq8Owned;
use vector::vect::VectOwned;
use vector::{VectorBorrowed, VectorOwned};

pub type Normalized = Vec<f32>;

pub trait Normalize: VectorOwned {
    fn normalize(vector: Self) -> Normalized;
    fn denormalize(x: Normalized) -> Self;
}

impl Normalize for VectOwned<f32> {
    fn normalize(vector: Self) -> Normalized {
        vector.into_vec()
    }

    fn denormalize(x: Normalized) -> Self {
        Self::new(x)
    }
}

impl Normalize for VectOwned<f16> {
    fn normalize(vector: Self) -> Normalized {
        f16::vector_to_f32(vector.slice())
    }

    fn denormalize(x: Normalized) -> Self {
        Self::new(f16::vector_from_f32(&x))
    }
}

impl Normalize for Rabitq8Owned {
    fn normalize(vector: Self) -> Normalized {
        let vector = vector.as_borrowed();
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dims() as _);
        for c in vector.code().iter().copied() {
            let base = -0.5 * ((1 << 8) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::rotate::rotate_reversed_inplace(&mut result);
        result
    }

    fn denormalize(x: Normalized) -> Self {
        let code = rabitq::byte::ugly_code(x.as_slice());
        Rabitq8Owned::new(
            code.0.dis_u_2,
            code.0.norm_of_lattice,
            code.0.sum_of_code,
            f32::reduce_sum_of_abs_x(&x),
            code.1,
        )
    }
}
