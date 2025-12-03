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

use crate::{VectorBorrowed, VectorOwned};
use distance::Distance;

#[derive(Debug, Clone)]
pub struct Rabitq8Owned {
    sum_of_x2: f32,
    norm_of_lattice: f32,
    sum_of_code: f32,
    sum_of_abs_x: f32,
    code: Vec<u8>,
}

impl Rabitq8Owned {
    #[inline(always)]
    pub fn new(
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        code: Vec<u8>,
    ) -> Self {
        Self::new_checked(sum_of_x2, norm_of_lattice, sum_of_code, sum_of_abs_x, code)
            .expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        code: Vec<u8>,
    ) -> Option<Self> {
        if !(1..=65535).contains(&code.len()) {
            return None;
        }
        #[allow(unsafe_code)]
        Some(unsafe {
            Self::new_unchecked(sum_of_x2, norm_of_lattice, sum_of_code, sum_of_abs_x, code)
        })
    }

    /// # Safety
    ///
    /// * `code.len()` must not be zero.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        code: Vec<u8>,
    ) -> Self {
        Self {
            sum_of_x2,
            norm_of_lattice,
            sum_of_code,
            sum_of_abs_x,
            code,
        }
    }
}

impl VectorOwned for Rabitq8Owned {
    type Borrowed<'a> = Rabitq8Borrowed<'a>;

    #[inline(always)]
    fn as_borrowed(&self) -> Rabitq8Borrowed<'_> {
        Rabitq8Borrowed {
            sum_of_x2: self.sum_of_x2,
            norm_of_lattice: self.norm_of_lattice,
            sum_of_code: self.sum_of_code,
            sum_of_abs_x: self.sum_of_abs_x,
            code: self.code.as_slice(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Rabitq8Borrowed<'a> {
    sum_of_x2: f32,
    norm_of_lattice: f32,
    sum_of_code: f32,
    sum_of_abs_x: f32,
    code: &'a [u8],
}

impl<'a> Rabitq8Borrowed<'a> {
    #[inline(always)]
    pub fn new(
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        code: &'a [u8],
    ) -> Self {
        Self::new_checked(sum_of_x2, norm_of_lattice, sum_of_code, sum_of_abs_x, code)
            .expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        code: &'a [u8],
    ) -> Option<Self> {
        if !(1..=65535).contains(&code.len()) {
            return None;
        }
        #[allow(unsafe_code)]
        Some(unsafe {
            Self::new_unchecked(sum_of_x2, norm_of_lattice, sum_of_code, sum_of_abs_x, code)
        })
    }

    /// # Safety
    ///
    /// * `code.len()` must not be zero.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        code: &'a [u8],
    ) -> Self {
        Self {
            sum_of_x2,
            norm_of_lattice,
            sum_of_code,
            sum_of_abs_x,
            code,
        }
    }

    #[inline(always)]
    pub fn sum_of_x2(&self) -> f32 {
        self.sum_of_x2
    }

    #[inline(always)]
    pub fn norm_of_lattice(&self) -> f32 {
        self.norm_of_lattice
    }

    #[inline(always)]
    pub fn sum_of_code(&self) -> f32 {
        self.sum_of_code
    }

    #[inline(always)]
    pub fn sum_of_abs_x(&self) -> f32 {
        self.sum_of_abs_x
    }

    #[inline(always)]
    pub fn code(&self) -> &'a [u8] {
        self.code
    }
}

impl VectorBorrowed for Rabitq8Borrowed<'_> {
    type Owned = Rabitq8Owned;

    #[inline(always)]
    fn dims(&self) -> u32 {
        self.code.len() as u32
    }

    #[inline(always)]
    fn own(&self) -> Rabitq8Owned {
        Rabitq8Owned {
            sum_of_x2: self.sum_of_x2,
            norm_of_lattice: self.norm_of_lattice,
            sum_of_code: self.sum_of_code,
            sum_of_abs_x: self.sum_of_abs_x,
            code: self.code.to_owned(),
        }
    }

    #[inline(always)]
    fn norm(&self) -> f32 {
        self.sum_of_x2.sqrt()
    }

    #[inline(always)]
    fn operator_dot(self, rhs: Self) -> Distance {
        assert_eq!(self.code.len(), rhs.code.len());
        let n = self.code.len() as u32;
        let value = rabitq::byte::binary::accumulate(self.code, rhs.code);
        Distance::from_f32(
            rabitq::byte::binary::half_process_dot(
                n,
                value,
                rabitq::byte::CodeMetadata {
                    dis_u_2: self.sum_of_x2,
                    norm_of_lattice: self.norm_of_lattice,
                    sum_of_code: self.sum_of_code,
                },
                rabitq::byte::CodeMetadata {
                    dis_u_2: rhs.sum_of_x2,
                    norm_of_lattice: rhs.norm_of_lattice,
                    sum_of_code: rhs.sum_of_code,
                },
            )
            .0,
        )
    }

    #[inline(always)]
    fn operator_l2s(self, rhs: Self) -> Distance {
        assert_eq!(self.code.len(), rhs.code.len());
        let n = self.code.len() as u32;
        let value = rabitq::byte::binary::accumulate(self.code, rhs.code);
        Distance::from_f32(
            rabitq::byte::binary::half_process_l2(
                n,
                value,
                rabitq::byte::CodeMetadata {
                    dis_u_2: self.sum_of_x2,
                    norm_of_lattice: self.norm_of_lattice,
                    sum_of_code: self.sum_of_code,
                },
                rabitq::byte::CodeMetadata {
                    dis_u_2: rhs.sum_of_x2,
                    norm_of_lattice: rhs.norm_of_lattice,
                    sum_of_code: rhs.sum_of_code,
                },
            )
            .0,
        )
    }

    #[inline(always)]
    fn operator_cos(self, rhs: Self) -> Distance {
        assert_eq!(self.code.len(), rhs.code.len());
        let n = self.code.len() as u32;
        let value = rabitq::byte::binary::accumulate(self.code, rhs.code);
        Distance::from_f32(
            rabitq::byte::binary::half_process_cos(
                n,
                value,
                rabitq::byte::CodeMetadata {
                    dis_u_2: self.sum_of_x2,
                    norm_of_lattice: self.norm_of_lattice,
                    sum_of_code: self.sum_of_code,
                },
                rabitq::byte::CodeMetadata {
                    dis_u_2: rhs.sum_of_x2,
                    norm_of_lattice: rhs.norm_of_lattice,
                    sum_of_code: rhs.sum_of_code,
                },
            )
            .0,
        )
    }

    #[inline(always)]
    fn operator_hamming(self, _: Self) -> Distance {
        unimplemented!()
    }

    #[inline(always)]
    fn operator_jaccard(self, _: Self) -> Distance {
        unimplemented!()
    }

    #[inline(always)]
    fn function_normalize(&self) -> Rabitq8Owned {
        Rabitq8Owned {
            sum_of_x2: 1.0,
            norm_of_lattice: self.norm_of_lattice,
            sum_of_code: self.sum_of_code,
            sum_of_abs_x: self.sum_of_abs_x / self.sum_of_x2.sqrt(),
            code: self.code.to_owned(),
        }
    }

    fn operator_add(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_sub(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_mul(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_and(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_or(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_xor(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }
}
