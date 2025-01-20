use crate::algorithm::operator::*;
use crate::algorithm::tuples::*;
use crate::algorithm::{Page, PageGuard, RelationRead, RelationWrite};
use crate::utils::pipe::Pipe;
use std::num::NonZeroU64;
use vector::VectorOwned;

pub fn vector_access_1<
    O: Operator,
    A: Accessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    relation: impl RelationRead,
    mean: IndexPointer,
    accessor: A,
) -> A::Output {
    let mut cursor = Err(mean);
    let mut result = accessor;
    while let Err(mean) = cursor.map_err(pointer_to_pair) {
        let vector_guard = relation.read(mean.0);
        let vector_tuple = vector_guard
            .get(mean.1)
            .expect("data corruption")
            .pipe(read_tuple::<VectorTuple<O::Vector>>);
        if vector_tuple.payload().is_some() {
            panic!("data corruption");
        }
        result.push(vector_tuple.elements());
        cursor = vector_tuple.metadata_or_pointer();
    }
    result.finish(cursor.expect("data corruption"))
}

pub fn vector_access_0<
    O: Operator,
    A: Accessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    relation: impl RelationRead,
    mean: IndexPointer,
    payload: NonZeroU64,
    accessor: A,
) -> Option<A::Output> {
    let mut cursor = Err(mean);
    let mut result = accessor;
    while let Err(mean) = cursor.map_err(pointer_to_pair) {
        let vector_guard = relation.read(mean.0);
        let vector_tuple = vector_guard
            .get(mean.1)?
            .pipe(read_tuple::<VectorTuple<O::Vector>>);
        if vector_tuple.payload().is_none() {
            panic!("data corruption");
        }
        if vector_tuple.payload() != Some(payload) {
            return None;
        }
        result.push(vector_tuple.elements());
        cursor = vector_tuple.metadata_or_pointer();
    }
    Some(result.finish(cursor.ok()?))
}

pub fn vector_append<O: Operator>(
    relation: impl RelationWrite + Clone,
    vectors_first: u32,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    payload: NonZeroU64,
) -> IndexPointer {
    fn append(relation: impl RelationWrite, first: u32, bytes: &[u8]) -> IndexPointer {
        if let Some(mut write) = relation.search(bytes.len()) {
            let i = write.alloc(bytes).unwrap();
            return pair_to_pointer((write.id(), i));
        }
        assert!(first != u32::MAX);
        let mut current = first;
        loop {
            let read = relation.read(current);
            if read.freespace() as usize >= bytes.len() || read.get_opaque().next == u32::MAX {
                drop(read);
                let mut write = relation.write(current, true);
                if let Some(i) = write.alloc(bytes) {
                    return pair_to_pointer((current, i));
                }
                if write.get_opaque().next == u32::MAX {
                    let mut extend = relation.extend(true);
                    write.get_opaque_mut().next = extend.id();
                    drop(write);
                    if let Some(i) = extend.alloc(bytes) {
                        let result = (extend.id(), i);
                        drop(extend);
                        let mut past = relation.write(first, true);
                        let skip = &mut past.get_opaque_mut().skip;
                        assert!(*skip != u32::MAX);
                        *skip = std::cmp::max(*skip, result.0);
                        return pair_to_pointer(result);
                    } else {
                        panic!("a tuple cannot even be fit in a fresh page");
                    }
                }
                if current == first && write.get_opaque().skip != first {
                    current = write.get_opaque().skip;
                } else {
                    current = write.get_opaque().next;
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
    let (metadata, slices) = O::Vector::vector_split(vector);
    let mut chain = Ok(metadata);
    for i in (0..slices.len()).rev() {
        chain = Err(append(
            relation.clone(),
            vectors_first,
            &serialize::<VectorTuple<O::Vector>>(&match chain {
                Ok(metadata) => VectorTuple::_0 {
                    elements: slices[i].to_vec(),
                    payload: Some(payload),
                    metadata,
                },
                Err(pointer) => VectorTuple::_1 {
                    elements: slices[i].to_vec(),
                    payload: Some(payload),
                    pointer,
                },
            }),
        ));
    }
    chain.err().unwrap()
}
