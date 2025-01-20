use std::num::NonZeroU64;

pub const fn pointer_to_ctid(pointer: NonZeroU64) -> pgrx::pg_sys::ItemPointerData {
    let value = pointer.get();
    pgrx::pg_sys::ItemPointerData {
        ip_blkid: pgrx::pg_sys::BlockIdData {
            bi_hi: ((value >> 32) & 0xffff) as u16,
            bi_lo: ((value >> 16) & 0xffff) as u16,
        },
        ip_posid: (value & 0xffff) as u16,
    }
}

pub const fn ctid_to_pointer(ctid: pgrx::pg_sys::ItemPointerData) -> NonZeroU64 {
    let mut value = 0;
    value |= (ctid.ip_blkid.bi_hi as u64) << 32;
    value |= (ctid.ip_blkid.bi_lo as u64) << 16;
    value |= ctid.ip_posid as u64;
    NonZeroU64::new(value).expect("invalid pointer")
}

#[allow(dead_code)]
const fn soundness_check(a: pgrx::pg_sys::ItemPointerData) {
    let b = ctid_to_pointer(a);
    let c = pointer_to_ctid(b);
    assert!(a.ip_blkid.bi_hi == c.ip_blkid.bi_hi);
    assert!(a.ip_blkid.bi_lo == c.ip_blkid.bi_lo);
    assert!(a.ip_posid == c.ip_posid);
}

const _: () = soundness_check(pgrx::pg_sys::ItemPointerData {
    ip_blkid: pgrx::pg_sys::BlockIdData { bi_hi: 1, bi_lo: 2 },
    ip_posid: 3,
});
