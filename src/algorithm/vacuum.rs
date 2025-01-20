use crate::algorithm::freepages;
use crate::algorithm::operator::Operator;
use crate::algorithm::tape::*;
use crate::algorithm::tuples::*;
use crate::algorithm::{Page, RelationWrite};
use crate::utils::pipe::Pipe;
use simd::fast_scan::unpack;
use std::num::NonZeroU64;

pub fn bulkdelete<O: Operator>(
    relation: impl RelationWrite,
    delay: impl Fn(),
    callback: impl Fn(NonZeroU64) -> bool,
) {
    let meta_guard = relation.read(0);
    let meta_tuple = meta_guard.get(1).unwrap().pipe(read_tuple::<MetaTuple>);
    let height_of_root = meta_tuple.height_of_root();
    let root_first = meta_tuple.root_first();
    let vectors_first = meta_tuple.vectors_first();
    drop(meta_guard);
    {
        type State = Vec<u32>;
        let mut state: State = vec![root_first];
        let step = |state: State| {
            let mut results = Vec::new();
            for first in state {
                let mut current = first;
                while current != u32::MAX {
                    let h1_guard = relation.read(current);
                    for i in 1..=h1_guard.len() {
                        let h1_tuple = h1_guard
                            .get(i)
                            .expect("data corruption")
                            .pipe(read_tuple::<H1Tuple>);
                        match h1_tuple {
                            H1TupleReader::_0(h1_tuple) => {
                                for first in h1_tuple.first().iter().copied() {
                                    results.push(first);
                                }
                            }
                            H1TupleReader::_1(_) => (),
                        }
                    }
                    current = h1_guard.get_opaque().next;
                }
            }
            results
        };
        for _ in (1..height_of_root).rev() {
            state = step(state);
        }
        for first in state {
            let jump_guard = relation.read(first);
            let jump_tuple = jump_guard
                .get(1)
                .expect("data corruption")
                .pipe(read_tuple::<JumpTuple>);
            let first = jump_tuple.first();
            let mut current = first;
            while current != u32::MAX {
                delay();
                let read = relation.read(current);
                let flag = 'flag: {
                    for i in 1..=read.len() {
                        let h0_tuple = read
                            .get(i)
                            .expect("data corruption")
                            .pipe(read_tuple::<H0Tuple>);
                        match h0_tuple {
                            H0TupleReader::_0(h0_tuple) => {
                                let p = h0_tuple.payload();
                                if let Some(payload) = p {
                                    if callback(payload) {
                                        break 'flag true;
                                    }
                                }
                            }
                            H0TupleReader::_1(h0_tuple) => {
                                let p = h0_tuple.payload();
                                for j in 0..32 {
                                    if let Some(payload) = p[j] {
                                        if callback(payload) {
                                            break 'flag true;
                                        }
                                    }
                                }
                            }
                            H0TupleReader::_2(_) => (),
                        }
                    }
                    false
                };
                if flag {
                    drop(read);
                    let mut write = relation.write(current, false);
                    for i in 1..=write.len() {
                        let h0_tuple = write
                            .get_mut(i)
                            .expect("data corruption")
                            .pipe(write_tuple::<H0Tuple>);
                        match h0_tuple {
                            H0TupleWriter::_0(mut h0_tuple) => {
                                let p = h0_tuple.payload();
                                if let Some(payload) = *p {
                                    if callback(payload) {
                                        *p = None;
                                    }
                                }
                            }
                            H0TupleWriter::_1(mut h0_tuple) => {
                                let p = h0_tuple.payload();
                                for j in 0..32 {
                                    if let Some(payload) = p[j] {
                                        if callback(payload) {
                                            p[j] = None;
                                        }
                                    }
                                }
                            }
                            H0TupleWriter::_2(_) => (),
                        }
                    }
                    current = write.get_opaque().next;
                } else {
                    current = read.get_opaque().next;
                }
            }
        }
    }
    {
        let first = vectors_first;
        let mut current = first;
        while current != u32::MAX {
            delay();
            let read = relation.read(current);
            let flag = 'flag: {
                for i in 1..=read.len() {
                    if let Some(vector_bytes) = read.get(i) {
                        let vector_tuple = vector_bytes.pipe(read_tuple::<VectorTuple<O::Vector>>);
                        let p = vector_tuple.payload();
                        if let Some(payload) = p {
                            if callback(payload) {
                                break 'flag true;
                            }
                        }
                    }
                }
                false
            };
            if flag {
                drop(read);
                let mut write = relation.write(current, true);
                for i in 1..=write.len() {
                    if let Some(vector_bytes) = write.get(i) {
                        let vector_tuple = vector_bytes.pipe(read_tuple::<VectorTuple<O::Vector>>);
                        let p = vector_tuple.payload();
                        if let Some(payload) = p {
                            if callback(payload) {
                                write.free(i);
                            }
                        }
                    };
                }
                current = write.get_opaque().next;
            } else {
                current = read.get_opaque().next;
            }
        }
    }
}

pub fn maintain<O: Operator>(relation: impl RelationWrite + Clone, delay: impl Fn()) {
    let meta_guard = relation.read(0);
    let meta_tuple = meta_guard.get(1).unwrap().pipe(read_tuple::<MetaTuple>);
    let dims = meta_tuple.dims();
    let height_of_root = meta_tuple.height_of_root();
    let root_first = meta_tuple.root_first();
    let freepage_first = meta_tuple.freepage_first();
    drop(meta_guard);

    let firsts = {
        type State = Vec<u32>;
        let mut state: State = vec![root_first];
        let step = |state: State| {
            let mut results = Vec::new();
            for first in state {
                let mut current = first;
                while current != u32::MAX {
                    delay();
                    let h1_guard = relation.read(current);
                    for i in 1..=h1_guard.len() {
                        let h1_tuple = h1_guard
                            .get(i)
                            .expect("data corruption")
                            .pipe(read_tuple::<H1Tuple>);
                        match h1_tuple {
                            H1TupleReader::_0(h1_tuple) => {
                                for first in h1_tuple.first().iter().copied() {
                                    results.push(first);
                                }
                            }
                            H1TupleReader::_1(_) => (),
                        }
                    }
                    current = h1_guard.get_opaque().next;
                }
            }
            results
        };
        for _ in (1..height_of_root).rev() {
            state = step(state);
        }
        state
    };

    for first in firsts {
        let mut jump_guard = relation.write(first, false);
        let mut jump_tuple = jump_guard
            .get_mut(1)
            .expect("data corruption")
            .pipe(write_tuple::<JumpTuple>);

        let mut tape = H0Tape::<_, _>::create(|| {
            if let Some(id) = freepages::fetch(relation.clone(), freepage_first) {
                let mut write = relation.write(id, false);
                write.clear();
                write
            } else {
                relation.extend(false)
            }
        });

        let mut trace = Vec::new();

        let first = *jump_tuple.first();
        let mut current = first;
        let mut computing = None;
        while current != u32::MAX {
            delay();
            trace.push(current);
            let h0_guard = relation.read(current);
            for i in 1..=h0_guard.len() {
                let h0_tuple = h0_guard
                    .get(i)
                    .expect("data corruption")
                    .pipe(read_tuple::<H0Tuple>);
                match h0_tuple {
                    H0TupleReader::_0(h0_tuple) => {
                        if let Some(payload) = h0_tuple.payload() {
                            tape.push(H0BranchWriter {
                                mean: h0_tuple.mean(),
                                dis_u_2: h0_tuple.code().0,
                                factor_ppc: h0_tuple.code().1,
                                factor_ip: h0_tuple.code().2,
                                factor_err: h0_tuple.code().3,
                                signs: h0_tuple
                                    .code()
                                    .4
                                    .iter()
                                    .flat_map(|x| {
                                        std::array::from_fn::<_, 64, _>(|i| *x & (1 << i) != 0)
                                    })
                                    .take(dims as _)
                                    .collect::<Vec<_>>(),
                                payload,
                            });
                        }
                    }
                    H0TupleReader::_1(h0_tuple) => {
                        let computing = &mut computing.take().unwrap_or_else(Vec::new);
                        computing.extend_from_slice(h0_tuple.elements());
                        let unpacked = unpack(computing);
                        for j in 0..32 {
                            if let Some(payload) = h0_tuple.payload()[j] {
                                tape.push(H0BranchWriter {
                                    mean: h0_tuple.mean()[j],
                                    dis_u_2: h0_tuple.metadata().0[j],
                                    factor_ppc: h0_tuple.metadata().1[j],
                                    factor_ip: h0_tuple.metadata().2[j],
                                    factor_err: h0_tuple.metadata().3[j],
                                    signs: unpacked[j]
                                        .iter()
                                        .flat_map(|&x| {
                                            [x & 1 != 0, x & 2 != 0, x & 4 != 0, x & 8 != 0]
                                        })
                                        .collect(),
                                    payload,
                                });
                            }
                        }
                    }
                    H0TupleReader::_2(h0_tuple) => {
                        let computing = computing.get_or_insert_with(Vec::new);
                        computing.extend_from_slice(h0_tuple.elements());
                    }
                }
            }
            current = h0_guard.get_opaque().next;
            drop(h0_guard);
        }

        let tape = tape.into_inner();
        let new = tape.first();
        drop(tape);

        *jump_tuple.first() = new;
        drop(jump_guard);

        freepages::mark(relation.clone(), freepage_first, &trace);
    }
}
