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
use vector::VectorOwned;
use vector::vect::VectOwned;

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
