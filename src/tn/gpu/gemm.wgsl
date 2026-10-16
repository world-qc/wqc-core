// Tiled complex f32 matrix product, with an independent batch.
// C[b, m, n] = A[b, m, k] * B[b, k, n]. Dimensions need not be multiples of 16.
// vec2f = (re, im). One workgroup writes a 16×16 tile of one batch item.

struct GemmParams {
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
}

@group(0) @binding(0) var<uniform> gemm_params: GemmParams;
@group(0) @binding(1) var<storage, read> gemm_a: array<vec2f>;
@group(0) @binding(2) var<storage, read> gemm_b: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> gemm_c: array<vec2f>;

var<workgroup> tile_a: array<vec2f, 256>;
var<workgroup> tile_b: array<vec2f, 256>;

@compute @workgroup_size(16, 16, 1)
fn gemm_f32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m = gemm_params.m;
    let kdim = gemm_params.k;
    let n = gemm_params.n;
    let batch = wg.z;
    if (batch >= gemm_params.batch) {
        return;
    }
    let a_off = batch * m * kdim;
    let b_off = batch * kdim * n;
    let c_off = batch * m * n;
    let row = wg.y * 16u + lid.y;
    let col = wg.x * 16u + lid.x;
    var acc = vec2f(0.0, 0.0);
    let tiles = (kdim + 15u) / 16u;
    for (var t: u32 = 0u; t < tiles; t = t + 1u) {
        let a_col = t * 16u + lid.x;
        let b_row = t * 16u + lid.y;
        var a_val = vec2f(0.0, 0.0);
        var b_val = vec2f(0.0, 0.0);
        if (row < m && a_col < kdim) {
            a_val = gemm_a[a_off + row * kdim + a_col];
        }
        if (b_row < kdim && col < n) {
            b_val = gemm_b[b_off + b_row * n + col];
        }
        tile_a[lid.y * 16u + lid.x] = a_val;
        tile_b[lid.y * 16u + lid.x] = b_val;
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let a_ik = tile_a[lid.y * 16u + kk];
            let b_kj = tile_b[kk * 16u + lid.x];
            acc = vec2f(
                acc.x + a_ik.x * b_kj.x - a_ik.y * b_kj.y,
                acc.y + a_ik.x * b_kj.y + a_ik.y * b_kj.x,
            );
        }
        workgroupBarrier();
    }
    if (row < m && col < n) {
        gemm_c[c_off + row * n + col] = acc;
    }
}
