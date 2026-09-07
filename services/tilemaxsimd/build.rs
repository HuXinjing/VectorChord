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
    if std::env::var_os("CARGO_FEATURE_BACKEND_METAL").is_some() {
        assert_eq!(
            std::env::var("CARGO_CFG_TARGET_OS").as_deref(),
            Ok("macos"),
            "backend-metal can only be built for macOS"
        );
        println!("cargo:rerun-if-changed=native/tilemaxsim_metal.mm");
        cc::Build::new()
            .cpp(true)
            .flag("-std=c++17")
            .flag("-fobjc-arc")
            .file("native/tilemaxsim_metal.mm")
            .compile("tilemaxsim_metal");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        return;
    }
    if std::env::var_os("CARGO_FEATURE_BACKEND_CUDA").is_none() {
        return;
    }
    println!("cargo:rerun-if-changed=native/tilemaxsim_cuda.cu");
    println!("cargo:rerun-if-env-changed=TILEMAXSIM_CUDA_ARCHS");
    let architectures = std::env::var("TILEMAXSIM_CUDA_ARCHS")
        .unwrap_or_else(|_| "80,89,90,90a,90-virtual".to_owned());
    let mut build = cc::Build::new();
    build.cuda(true).flag("-O3").flag("-lineinfo");
    for architecture in architectures
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        let (target, code) = architecture
            .strip_suffix("-virtual")
            .map_or((architecture, format!("sm_{architecture}")), |target| {
                (target, format!("compute_{target}"))
            });
        let number = target.trim_end_matches(['a', 'f']);
        assert!(
            !number.is_empty()
                && number.bytes().all(|byte| byte.is_ascii_digit())
                && target
                    .strip_prefix(number)
                    .is_some_and(|suffix| matches!(suffix, "" | "a" | "f")),
            "TILEMAXSIM_CUDA_ARCHS contains an invalid architecture"
        );
        build.flag(format!("-gencode=arch=compute_{target},code={code}"));
    }
    build
        .file("native/tilemaxsim_cuda.cu")
        .compile("tilemaxsim_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
