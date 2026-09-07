// This software is licensed under the repository's dual license model.

use anyhow::Result;

#[cfg(not(any(
    feature = "backend-cuda",
    feature = "backend-cpu",
    feature = "backend-metal"
)))]
compile_error!("the tilemaxsimd executable requires one accelerator backend");

fn main() -> Result<()> {
    #[cfg(feature = "backend-cpu")]
    return tilemaxsimd::daemon::run_with_backend_factory(|device, total, workspace| {
        Ok(Box::new(tilemaxsimd::cpu::CpuBackend::create(
            device, total, workspace,
        )?))
    });

    #[cfg(feature = "backend-cuda")]
    return tilemaxsimd::daemon::run_with_backend_factory(|device, total, workspace| {
        Ok(Box::new(tilemaxsimd::gpu::Gpu::create(
            device, total, workspace,
        )?))
    });

    #[cfg(feature = "backend-metal")]
    return tilemaxsimd::daemon::run_with_backend_factory(|device, total, workspace| {
        Ok(Box::new(tilemaxsimd::metal::MetalBackend::create(
            device, total, workspace,
        )?))
    });
}
