use crate::operator::FunctionalAccessor;
use crate::tuples::{MetaTuple, WithReader};
use crate::{Page, RelationRead, tape};

pub fn cache(index: impl RelationRead) -> Vec<u32> {
    let mut trace = vec![0_u32];
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let height_of_root = meta_tuple.height_of_root();
    let root_first = meta_tuple.root_first();
    drop(meta_guard);
    type State = Vec<u32>;
    let mut state: State = vec![root_first];
    let mut step = |state: State| {
        let mut results = Vec::new();
        for first in state {
            tape::read_h1_tape(
                index.clone(),
                first,
                || {
                    fn push<T>(_: &mut (), _: &[T]) {}
                    fn finish<T>(_: (), _: (&T, &T, &T, &T)) -> [(); 32] {
                        [(); 32]
                    }
                    FunctionalAccessor::new((), push, finish)
                },
                |(), _, first| {
                    results.push(first);
                },
                |id| {
                    trace.push(id);
                },
            );
        }
        results
    };
    for _ in (1..height_of_root).rev() {
        state = step(state);
    }
    for first in state {
        trace.push(first);
    }
    trace
}
