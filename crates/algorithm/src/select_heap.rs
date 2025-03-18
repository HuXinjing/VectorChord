use std::collections::BinaryHeap;
use std::num::NonZero;

pub enum SelectHeap<T> {
    Sorted { x: Vec<T>, t: NonZero<usize> },
    Heap { x: BinaryHeap<T> },
}

impl<T: Ord> SelectHeap<T> {
    pub fn from_vec(mut vec: Vec<T>) -> Self {
        let n = vec.len();
        if let Some(t) = NonZero::new(n / 384) {
            let index = n - t.get();
            turboselect::select_nth_unstable(&mut vec, index);
            vec[index..].sort_unstable();
            Self::Sorted { x: vec, t }
        } else {
            Self::Heap {
                x: BinaryHeap::from(vec),
            }
        }
    }
    pub fn is_empty(&self) -> bool {
        match self {
            SelectHeap::Sorted { x, .. } => x.is_empty(),
            SelectHeap::Heap { x } => x.is_empty(),
        }
    }
    pub fn pop(&mut self) -> Option<T> {
        match self {
            SelectHeap::Sorted { x: inner, t } => {
                let x = inner.pop().expect("inconsistent internal structure");
                if let Some(value) = NonZero::new(t.get() - 1) {
                    *t = value;
                } else {
                    *self = SelectHeap::Heap {
                        x: std::mem::take(inner).into(),
                    };
                }
                Some(x)
            }
            SelectHeap::Heap { x } => x.pop(),
        }
    }
    pub fn peek(&self) -> Option<&T> {
        match self {
            SelectHeap::Sorted { x, .. } => x.last(),
            SelectHeap::Heap { x } => x.peek(),
        }
    }
}

#[test]
fn test_select_heap() {
    for _ in 0..1000 {
        let sequence = (0..10000)
            .map(|_| rand::random::<i32>())
            .collect::<Vec<_>>();
        let answer = {
            let mut x = sequence.clone();
            x.sort_by_key(|x| std::cmp::Reverse(*x));
            x
        };
        let result = {
            let mut x = SelectHeap::from_vec(sequence.clone());
            std::iter::from_fn(|| x.pop()).collect::<Vec<_>>()
        };
        assert_eq!(answer, result);
    }
}

#[test]
fn test_issue_209() {
    let mut heap = SelectHeap::from_vec(vec![0]);
    assert_eq!(heap.pop(), Some(0));
    assert_eq!(heap.pop(), None);
}
