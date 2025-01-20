use crate::algorithm::operator::*;
use crate::algorithm::tape::read_h1_tape;
use crate::algorithm::tuples::*;
use crate::algorithm::vectors::{self};
use crate::algorithm::{Page, PageGuard, RelationWrite};
use crate::utils::pipe::Pipe;
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroU64;
use vector::VectorBorrowed;
use vector::VectorOwned;

pub fn insert<O: Operator>(
    relation: impl RelationWrite + Clone,
    payload: NonZeroU64,
    vector: O::Vector,
) {
    let vector = O::Vector::random_projection(vector.as_borrowed());
    let meta_guard = relation.read(0);
    let meta_tuple = meta_guard.get(1).unwrap().pipe(read_tuple::<MetaTuple>);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let height_of_root = meta_tuple.height_of_root();
    assert_eq!(dims, vector.as_borrowed().dims(), "unmatched dimensions");
    let root_mean = meta_tuple.root_mean();
    let root_first = meta_tuple.root_first();
    let vectors_first = meta_tuple.vectors_first();
    drop(meta_guard);

    let default_lut_block = if !is_residual {
        Some(O::Vector::compute_lut_block(vector.as_borrowed()))
    } else {
        None
    };

    let mean = vectors::vector_append::<O>(
        relation.clone(),
        vectors_first,
        vector.as_borrowed(),
        payload,
    );

    type State<O> = (u32, Option<<O as Operator>::Vector>);
    let mut state: State<O> = {
        let mean = root_mean;
        if is_residual {
            let residual_u = vectors::vector_access_1::<O, _>(
                relation.clone(),
                mean,
                LAccess::new(
                    O::Vector::elements_and_metadata(vector.as_borrowed()),
                    O::ResidualAccessor::default(),
                ),
            );
            (root_first, Some(residual_u))
        } else {
            (root_first, None)
        }
    };
    let step = |state: State<O>| {
        let mut results = Vec::new();
        {
            let (first, residual) = state;
            let lut = if let Some(residual) = residual {
                &O::Vector::compute_lut_block(residual.as_borrowed())
            } else {
                default_lut_block.as_ref().unwrap()
            };
            read_h1_tape(
                relation.clone(),
                first,
                || {
                    RAccess::new(
                        (&lut.4, (lut.0, lut.1, lut.2, lut.3, 1.9f32)),
                        O::Distance::block_accessor(),
                    )
                },
                |lowerbound, mean, first| {
                    results.push((Reverse(lowerbound), AlwaysEqual(mean), AlwaysEqual(first)));
                },
            );
        }
        let mut heap = BinaryHeap::from(results);
        let mut cache = BinaryHeap::<(Reverse<Distance>, _, _)>::new();
        {
            while !heap.is_empty() && heap.peek().map(|x| x.0) > cache.peek().map(|x| x.0) {
                let (_, AlwaysEqual(mean), AlwaysEqual(first)) = heap.pop().unwrap();
                if is_residual {
                    let (dis_u, residual_u) = vectors::vector_access_1::<O, _>(
                        relation.clone(),
                        mean,
                        LAccess::new(
                            O::Vector::elements_and_metadata(vector.as_borrowed()),
                            (
                                O::DistanceAccessor::default(),
                                O::ResidualAccessor::default(),
                            ),
                        ),
                    );
                    cache.push((
                        Reverse(dis_u),
                        AlwaysEqual(first),
                        AlwaysEqual(Some(residual_u)),
                    ));
                } else {
                    let dis_u = vectors::vector_access_1::<O, _>(
                        relation.clone(),
                        mean,
                        LAccess::new(
                            O::Vector::elements_and_metadata(vector.as_borrowed()),
                            O::DistanceAccessor::default(),
                        ),
                    );
                    cache.push((Reverse(dis_u), AlwaysEqual(first), AlwaysEqual(None)));
                }
            }
            let (_, AlwaysEqual(first), AlwaysEqual(mean)) = cache.pop().unwrap();
            (first, mean)
        }
    };
    for _ in (1..height_of_root).rev() {
        state = step(state);
    }

    let (first, residual) = state;
    let code = if let Some(residual) = residual {
        O::Vector::code(residual.as_borrowed())
    } else {
        O::Vector::code(vector.as_borrowed())
    };
    let bytes = serialize(&H0Tuple::_0 {
        mean,
        dis_u_2: code.dis_u_2,
        factor_ppc: code.factor_ppc,
        factor_ip: code.factor_ip,
        factor_err: code.factor_err,
        payload: Some(payload),
        elements: rabitq::pack_to_u64(&code.signs),
    });

    let jump_guard = relation.read(first);
    let jump_tuple = jump_guard
        .get(1)
        .expect("data corruption")
        .pipe(read_tuple::<JumpTuple>);

    let first = jump_tuple.first();

    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = relation.read(current);
        if read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = relation.write(current, false);
            if write.get_opaque().next == u32::MAX {
                if write.alloc(&bytes).is_some() {
                    return;
                }
                let mut extend = relation.extend(false);
                write.get_opaque_mut().next = extend.id();
                drop(write);
                let fresh = extend.id();
                if extend.alloc(&bytes).is_some() {
                    drop(extend);
                    let mut past = relation.write(first, false);
                    past.get_opaque_mut().skip = std::cmp::max(past.get_opaque_mut().skip, fresh);
                    drop(past);
                    return;
                } else {
                    panic!("a tuple cannot even be fit in a fresh page");
                }
            } else {
                if current == first && write.get_opaque().skip != first {
                    current = write.get_opaque().skip;
                } else {
                    current = write.get_opaque().next;
                }
            }
        } else {
            if current == first && read.get_opaque().skip != first {
                current = read.get_opaque().skip;
            } else {
                current = read.get_opaque().next;
            }
        }
    }
}
