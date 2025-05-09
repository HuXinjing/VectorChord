use crate::tuples::{MetaTuple, WithReader};
use crate::{Page, RelationRead};

pub struct Cost {
    pub dims: u32,
    pub is_residual: bool,
    pub cells: Vec<u32>,
}

#[must_use]
pub fn cost<R: RelationRead>(index: &R) -> Cost {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let cells = meta_tuple.cells().to_vec();
    drop(meta_guard);

    Cost {
        dims,
        is_residual,
        cells,
    }
}
