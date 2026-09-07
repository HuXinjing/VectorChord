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
    println!("cargo:rerun-if-env-changed=TILEMAXSIM_CUDA_ARCHS");
    let architectures =
        std::env::var("TILEMAXSIM_CUDA_ARCHS").unwrap_or_else(|_| "80,89,90,90-virtual".to_owned());
    let mut build = cc::Build::new();
    build.cuda(true).flag("-O3").flag("-lineinfo");
    for architecture in architectures
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        let (number, code) = architecture
            .strip_suffix("-virtual")
            .map_or((architecture, format!("sm_{architecture}")), |number| {
                (number, format!("compute_{number}"))
            });
        assert!(
            number.bytes().all(|byte| byte.is_ascii_digit()),
            "TILEMAXSIM_CUDA_ARCHS contains an invalid architecture"
        );
        build.flag(&format!("-gencode=arch=compute_{number},code={code}"));
    }
    build
        .file("native/tilemaxsim_cuda.cu")
        .compile("tilemaxsim_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
