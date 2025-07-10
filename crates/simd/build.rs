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

use std::env::var;
use std::error::Error;
use std::ffi::OsString;

fn compiler(host: &str, target: &str) -> Option<OsString> {
    if let Some(cc) = std::env::var_os("CC") {
        return Some(cc);
    }
    if let Some(cc) = std::env::var_os(format!("CC_{target}")) {
        return Some(cc);
    }
    if let Some(cc) = std::env::var_os(format!("CC_{}", target.replace("-", "_"))) {
        return Some(cc);
    }
    if host == target {
        if let Ok(cc) = which::which("clang") {
            return Some(cc.into());
        }
        for i in (16..=99).rev() {
            if let Ok(cc) = which::which(format!("clang-{i}")) {
                return Some(cc.into());
            }
        }
    }
    None
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo::rerun-if-changed=cshim");
    let host = var("HOST")?;
    let target = var("TARGET")?;
    let target_arch = var("CARGO_CFG_TARGET_ARCH")?;
    match target_arch.as_str() {
        "aarch64" => {
            let mut build = cc::Build::new();
            if let Some(compiler) = compiler(&host, &target) {
                build.compiler(compiler);
            }
            build.file("./cshim/aarch64.c");
            build.opt_level(3);
            build.compile("simd_cshim");
        }
        "x86_64" => {
            let mut build = cc::Build::new();
            if let Some(compiler) = compiler(&host, &target) {
                build.compiler(compiler);
            }
            build.file("./cshim/x86_64.c");
            build.opt_level(3);
            build.compile("simd_cshim");
        }
        _ => (),
    }
    Ok(())
}
