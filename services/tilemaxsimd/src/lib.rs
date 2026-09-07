// This software is licensed under the repository's dual license model.

pub mod backend;
pub mod cache;
#[cfg(feature = "backend-cpu")]
pub mod cpu;
pub mod daemon;
pub mod dispatch;
pub mod engine;
#[cfg(feature = "backend-cuda")]
pub mod gpu;
pub mod protocol;
pub mod quant;
pub mod scheduler;
pub mod shard;

#[cfg(all(feature = "backend-cuda", feature = "backend-cpu"))]
compile_error!("select exactly one accelerator backend");
