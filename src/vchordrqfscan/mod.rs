mod algorithm;
mod gucs;
mod index;
mod types;

pub unsafe fn init() {
    unsafe {
        index::init();
        gucs::init();
    }
}
