// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// Copyright (c) 2026 Hu Xinjing

fn main() {
    println!("cargo:rerun-if-changed=native/tilemaxsim_cuda.cu");
    cc::Build::new()
        .cuda(true)
        .flag("-O3")
        .flag("-lineinfo")
        .file("native/tilemaxsim_cuda.cu")
        .compile("tilemaxsim_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
