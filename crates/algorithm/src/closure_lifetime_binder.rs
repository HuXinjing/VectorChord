// Use stable language features as an alternative to `closure_lifetime_binder`.
// See https://github.com/rust-lang/rust/issues/97362.

#[inline(always)]
pub fn id_0<F, A: ?Sized, B: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(&'a mut A, &'a B) -> R,
{
    f
}

#[inline(always)]
pub fn id_1<F, A: ?Sized, B: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(A, (&'a B, &'a B, &'a B, &'a B)) -> R,
{
    f
}

#[inline(always)]
pub fn id_2<F, A: ?Sized, B: ?Sized, C: ?Sized, D: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(A, B, C, &'a D) -> R,
{
    f
}

#[inline(always)]
pub fn id_3<F, T, A: ?Sized>(f: F) -> F
where
    T: crate::RelationWrite,
    F: for<'a> Fn(&'a T, A) -> T::WriteGuard<'a>,
{
    f
}

#[inline(always)]
pub fn id_4<F, T, A: ?Sized, B: ?Sized, R: ?Sized>(f: F) -> F
where
    T: crate::RelationRead,
    F: FnMut(A, Vec<T::ReadGuard<'_>>, B) -> R,
{
    f
}
