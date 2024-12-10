use crate::datatype::memory_pgvector_halfvec::PgvectorHalfvecInput;
use crate::datatype::memory_pgvector_vector::PgvectorVectorInput;
use crate::datatype::memory_scalar8::Scalar8Output;
use crate::types::scalar8::Scalar8Borrowed;
use base::simd::ScalarLike;
use half::f16;

#[pgrx::pg_extern(sql = "")]
fn _vchord_vector_quantize_to_scalar8(vector: PgvectorVectorInput) -> Scalar8Output {
    let vector = vector.as_borrowed();
    let sum_of_x2 = f32::reduce_sum_of_x2(vector.slice());
    let (k, b, code) =
        base::simd::quantize::quantize(f32::vector_to_f32_borrowed(vector.slice()).as_ref(), 255.0);
    let sum_of_code = base::simd::u8::reduce_sum_of_x_as_u32(&code) as f32;
    Scalar8Output::new(Scalar8Borrowed::new(sum_of_x2, k, b, sum_of_code, &code))
}

#[pgrx::pg_extern(sql = "")]
fn _vchord_halfvec_quantize_to_scalar8(vector: PgvectorHalfvecInput) -> Scalar8Output {
    let vector = vector.as_borrowed();
    let sum_of_x2 = f16::reduce_sum_of_x2(vector.slice());
    let (k, b, code) =
        base::simd::quantize::quantize(f16::vector_to_f32_borrowed(vector.slice()).as_ref(), 255.0);
    let sum_of_code = base::simd::u8::reduce_sum_of_x_as_u32(&code) as f32;
    Scalar8Output::new(Scalar8Borrowed::new(sum_of_x2, k, b, sum_of_code, &code))
}
