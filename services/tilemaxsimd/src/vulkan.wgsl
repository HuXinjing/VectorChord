// Exact MaxSim: one workgroup per (candidate, query row), FP32 accumulation.
// FP16 is unpacked from u32 words; shader-f16 hardware is not required.
struct Params { query_rows: u32, dimension: u32, dtype: u32, candidates: u32 }
@group(0) @binding(0) var<storage, read> documents: array<u32>;
@group(0) @binding(1) var<storage, read> query: array<u32>;
@group(0) @binding(2) var<storage, read> metadata: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;
var<workgroup> partial: array<f32, 64>;
fn doc_value(index: u32) -> f32 {
    if params.dtype == 1u { return bitcast<f32>(documents[index]); }
    return unpack2x16float(documents[index / 2u])[index % 2u];
}
fn query_value(index: u32) -> f32 {
    if params.dtype == 1u { return bitcast<f32>(query[index]); }
    return unpack2x16float(query[index / 2u])[index % 2u];
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let candidate = group.x;
    let q = group.y;
    let scalar_bytes = select(2u, 4u, params.dtype == 1u);
    let base = metadata[candidate].x / scalar_bytes;
    var maximum = bitcast<f32>(0xff800000u);
    for (var row = 0u; row < metadata[candidate].y; row++) {
        var dot = 0.0;
        for (var d = lane; d < params.dimension; d += 64u) {
            dot += query_value(q * params.dimension + d) * doc_value(base + row * params.dimension + d);
        }
        partial[lane] = dot;
        workgroupBarrier();
        for (var stride = 32u; stride > 0u; stride /= 2u) {
            if lane < stride { partial[lane] += partial[lane + stride]; }
            workgroupBarrier();
        }
        maximum = max(maximum, partial[0]);
        workgroupBarrier();
    }
    if lane == 0u { scores[candidate * params.query_rows + q] = maximum; }
}
