use crate::algorithm::operator::Operator;
use crate::algorithm::tuples::*;
use crate::algorithm::vectors;
use crate::algorithm::{Page, RelationRead};
use crate::utils::pipe::Pipe;
use std::fmt::Write;

pub fn prewarm<O: Operator>(relation: impl RelationRead + Clone, height: i32) -> String {
    let meta_guard = relation.read(0);
    let meta_tuple = meta_guard.get(1).unwrap().pipe(read_tuple::<MetaTuple>);
    let height_of_root = meta_tuple.height_of_root();
    let root_mean = meta_tuple.root_mean();
    let root_first = meta_tuple.root_first();
    drop(meta_guard);

    let mut message = String::new();
    writeln!(message, "height of root: {}", height_of_root).unwrap();
    let prewarm_max_height = if height < 0 { 0 } else { height as u32 };
    if prewarm_max_height > height_of_root {
        return message;
    }
    type State = Vec<u32>;
    let mut state: State = {
        let mut results = Vec::new();
        let counter = 1_usize;
        {
            vectors::vector_access_1::<O, _>(relation.clone(), root_mean, ());
            results.push(root_first);
        }
        writeln!(message, "number of tuples: {}", results.len()).unwrap();
        writeln!(message, "number of pages: {}", counter).unwrap();
        results
    };
    let mut step = |state: State| {
        let mut counter = 0_usize;
        let mut results = Vec::new();
        for list in state {
            let mut current = list;
            while current != u32::MAX {
                counter += 1;
                pgrx::check_for_interrupts!();
                let h1_guard = relation.read(current);
                for i in 1..=h1_guard.len() {
                    let h1_tuple = h1_guard
                        .get(i)
                        .expect("data corruption")
                        .pipe(read_tuple::<H1Tuple>);
                    match h1_tuple {
                        H1TupleReader::_0(h1_tuple) => {
                            for mean in h1_tuple.mean().iter().copied() {
                                vectors::vector_access_1::<O, _>(relation.clone(), mean, ());
                            }
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
        writeln!(message, "number of tuples: {}", results.len()).unwrap();
        writeln!(message, "number of pages: {}", counter).unwrap();
        results
    };
    for _ in (std::cmp::max(1, prewarm_max_height)..height_of_root).rev() {
        state = step(state);
    }
    if prewarm_max_height == 0 {
        let mut counter = 0_usize;
        let mut results = Vec::new();
        for list in state {
            let jump_guard = relation.read(list);
            let jump_tuple = jump_guard
                .get(1)
                .expect("data corruption")
                .pipe(read_tuple::<JumpTuple>);
            let first = jump_tuple.first();
            let mut current = first;
            while current != u32::MAX {
                counter += 1;
                pgrx::check_for_interrupts!();
                let h0_guard = relation.read(current);
                for i in 1..=h0_guard.len() {
                    let _h0_tuple = h0_guard
                        .get(i)
                        .expect("data corruption")
                        .pipe(read_tuple::<H0Tuple>);
                    results.push(());
                }
                current = h0_guard.get_opaque().next;
            }
        }
        writeln!(message, "number of tuples: {}", results.len()).unwrap();
        writeln!(message, "number of pages: {}", counter).unwrap();
    }
    message
}
