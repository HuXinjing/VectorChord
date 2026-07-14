// Copyright (c) 2026 HuXinjing

fn main() {
    println!("cargo:rerun-if-changed=native/tilemaxsim_cuda.cu");
    cc::Build::new()
        .cuda(true)
        .debug(false)
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-lineinfo")
        .file("native/tilemaxsim_cuda.cu")
        .compile("tilemaxsim_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
