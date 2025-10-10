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

use index::prefetcher::Sequence;

pub struct Filter<S, P> {
    sequence: S,
    predicate: P,
}

impl<S, P> Sequence for Filter<S, P>
where
    S: Sequence,
    P: FnMut(&S::Item) -> bool,
{
    type Item = S::Item;

    type Inner = S::Inner;

    fn next(&mut self) -> Option<Self::Item> {
        while !(self.predicate)(self.sequence.peek()?) {
            let _ = self.sequence.next();
        }
        self.sequence.next()
    }

    fn peek(&mut self) -> Option<&Self::Item> {
        while !(self.predicate)(self.sequence.peek()?) {
            let _ = self.sequence.next();
        }
        self.sequence.peek()
    }

    fn into_inner(self) -> Self::Inner {
        self.sequence.into_inner()
    }
}

pub fn filter<S, P>(sequence: S, predicate: P) -> Filter<S, P> {
    Filter {
        sequence,
        predicate,
    }
}
