use anyhow::{Result, anyhow, bail};
use std::ffi::{CStr, c_char, c_int, c_uchar, c_void};
use std::ptr::NonNull;

#[repr(C)]
struct NativeGpu(c_void);

unsafe extern "C" {
    fn vctm_gpu_create(
        device: c_int,
        total_bytes: usize,
        workspace_bytes: usize,
        output: *mut *mut NativeGpu,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn vctm_gpu_destroy(gpu: *mut NativeGpu);
    fn vctm_gpu_tensor_bytes(gpu: *const NativeGpu) -> usize;
    fn vctm_gpu_upload_batch(
        gpu: *mut NativeGpu,
        offsets: *const u64,
        payloads: *const *const c_uchar,
        lengths: *const usize,
        count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn vctm_gpu_score(
        gpu: *mut NativeGpu,
        query: *const c_uchar,
        query_bytes: usize,
        query_rows: u32,
        dimension: u32,
        dtype: u8,
        document_offsets: *const u64,
        document_rows: *const u32,
        count: usize,
        output: *mut f32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
}

pub struct Gpu {
    native: NonNull<NativeGpu>,
    device: i32,
    tensor_bytes: usize,
}

// SAFETY: `Gpu` uniquely owns the native handle. It may move to a scoped
// worker, but no method exposes the pointer and all calls require `&mut self`.
unsafe impl Send for Gpu {}

impl Gpu {
    pub fn create(device: i32, total_bytes: usize, workspace_bytes: usize) -> Result<Self> {
        let mut native = std::ptr::null_mut();
        let mut error = [0_i8; 512];
        // SAFETY: the C API writes one opaque pointer and a bounded error string.
        let status = unsafe {
            vctm_gpu_create(
                device,
                total_bytes,
                workspace_bytes,
                &mut native,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!(native_error(&error));
        }
        let native = NonNull::new(native).ok_or_else(|| anyhow!("CUDA returned a null arena"))?;
        // SAFETY: `native` is live until Drop.
        let tensor_bytes = unsafe { vctm_gpu_tensor_bytes(native.as_ptr()) };
        Ok(Self {
            native,
            device,
            tensor_bytes,
        })
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn tensor_bytes(&self) -> usize {
        self.tensor_bytes
    }

    pub fn upload_batch(&mut self, items: &[(u64, &[u8])]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let offsets = items.iter().map(|item| item.0).collect::<Vec<_>>();
        let payloads = items.iter().map(|item| item.1.as_ptr()).collect::<Vec<_>>();
        let lengths = items.iter().map(|item| item.1.len()).collect::<Vec<_>>();
        let mut error = [0_i8; 512];
        // SAFETY: all slices remain alive for the synchronous native batch call.
        let status = unsafe {
            vctm_gpu_upload_batch(
                self.native.as_ptr(),
                offsets.as_ptr(),
                payloads.as_ptr(),
                lengths.as_ptr(),
                items.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!(native_error(&error));
        }
        Ok(())
    }

    pub fn score(
        &mut self,
        query: &[u8],
        query_rows: u32,
        dimension: u32,
        dtype: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<f32>> {
        if document_offsets.len() != document_rows.len() || document_offsets.is_empty() {
            bail!("invalid native document metadata");
        }
        let mut output = vec![0.0_f32; document_offsets.len()];
        let mut error = [0_i8; 512];
        // SAFETY: the native call is synchronous and receives valid slice pointers.
        let status = unsafe {
            vctm_gpu_score(
                self.native.as_ptr(),
                query.as_ptr(),
                query.len(),
                query_rows,
                dimension,
                dtype,
                document_offsets.as_ptr(),
                document_rows.as_ptr(),
                document_offsets.len(),
                output.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!(native_error(&error));
        }
        if output.iter().any(|score| !score.is_finite()) {
            bail!("native TileMaxSim returned a non-finite score");
        }
        Ok(output)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: this is the unique owned native pointer.
        unsafe { vctm_gpu_destroy(self.native.as_ptr()) };
    }
}

fn native_error(buffer: &[c_char]) -> String {
    // SAFETY: the native helper always NUL-terminates a nonempty error buffer.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}
