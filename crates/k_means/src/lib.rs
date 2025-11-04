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

pub mod flat;
pub mod hierarchical;
pub mod quick;
pub mod rabitq;
pub mod square;

use crate::square::{Square, SquareMut};
use rand::rngs::StdRng;

pub struct This<'a> {
    pool: &'a rayon::ThreadPool,
    rng: StdRng,
    d: usize,
    c: usize,
    centroids: Square,
    targets: Vec<usize>,
}

pub trait KMeans<'a> {
    fn this(&mut self) -> &mut This<'a>;
    fn assign(&mut self);
    fn update(&mut self);
    fn sphericalize(&mut self) {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        use simd::Floating;
        let this = self.this();
        this.pool.install(|| {
            (&mut this.centroids).into_par_iter().for_each(|centroid| {
                let l = f32::reduce_sum_of_x2(centroid).sqrt();
                f32::vector_mul_scalar_inplace(centroid, 1.0 / l);
            });
        });
    }
    fn index(&mut self) -> Box<dyn Fn(&[f32]) -> u32 + '_>;
    fn finish(self: Box<Self>) -> Square;
}

pub fn k_means<'a>(
    pool: &'a rayon::ThreadPool,
    d: usize,
    samples: SquareMut<'a>,
    c: usize,
    seed: [u8; 32],
) -> Box<dyn KMeans<'a> + 'a> {
    assert!(d > 0 && c > 0);
    let n = samples.len();
    if n <= c {
        quick::new(pool, d, samples, c, seed)
    } else if n <= c * 2 {
        flat::new(pool, d, samples, c, seed)
    } else {
        rabitq::new(pool, d, samples, c, seed)
    }
}

pub fn hierarchical_k_means<'a>(
    pool: &'a rayon::ThreadPool,
    d: usize,
    samples: SquareMut<'a>,
    c: usize,
    seed: [u8; 32],
    is_spherical: bool,
) -> Box<dyn KMeans<'a> + 'a> {
    assert!(d > 0 && c > 0);
    hierarchical::new(pool, d, samples, c, seed, is_spherical)
}
