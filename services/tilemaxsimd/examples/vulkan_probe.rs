// This software is licensed under the repository's dual license model.
use anyhow::Result;
use tilemaxsimd::backend::{AcceleratorBackend, run_conformance_probe};
use tilemaxsimd::vulkan::VulkanBackend;

fn main() -> Result<()> {
    let mut backend = VulkanBackend::create(0, 128 << 20, 32 << 20)?;
    let report = run_conformance_probe(&mut backend)?;
    println!(
        "{}",
        serde_json::json!({"device": backend.info(), "conformance": report})
    );
    Ok(())
}
