// MPS tensor kernels — complex numbers as vec2<f32> (re, im).

fn cmplx_mul(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cmplx_add(a: vec2f, b: vec2f) -> vec2f {
    return a + b;
}

struct OneQubitParams {
    left: u32,
    right: u32,
    _pad0: u32,
    _pad1: u32,
    u00: vec2f,
    u01: vec2f,
    u10: vec2f,
    u11: vec2f,
}

@group(0) @binding(0) var<uniform> one_qubit_params: OneQubitParams;
@group(0) @binding(1) var<storage, read_write> site_buf: array<vec2f>;

@compute @workgroup_size(256)
fn apply_one_qubit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pairs = one_qubit_params.left * one_qubit_params.right;
    let idx = gid.x;
    if (idx >= pairs) {
        return;
    }
    let b = idx % one_qubit_params.right;
    let a = idx / one_qubit_params.right;
    let stride = one_qubit_params.right;
    let base = a * 2u * stride + b;
    let v0 = site_buf[base];
    let v1 = site_buf[base + stride];
    site_buf[base] = cmplx_add(
        cmplx_mul(one_qubit_params.u00, v0),
        cmplx_mul(one_qubit_params.u01, v1),
    );
    site_buf[base + stride] = cmplx_add(
        cmplx_mul(one_qubit_params.u10, v0),
        cmplx_mul(one_qubit_params.u11, v1),
    );
}

struct MergeParams {
    dl: u32,
    dr: u32,
    bond: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> merge_params: MergeParams;
@group(0) @binding(1) var<storage, read> left_buf: array<vec2f>;
@group(0) @binding(2) var<storage, read> right_buf: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> theta_buf: array<vec2f>;

@compute @workgroup_size(256)
fn merge_two_site(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dl = merge_params.dl;
    let dr = merge_params.dr;
    let bond = merge_params.bond;
    let total = dl * 2u * 2u * dr;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let r = idx % dr;
    let t = (idx / dr) % 2u;
    let s = (idx / (dr * 2u)) % 2u;
    let a = idx / (dr * 2u * 2u);

    var sum = vec2f(0.0, 0.0);
    for (var g: u32 = 0u; g < bond; g = g + 1u) {
        let l_idx = a * 2u * bond + s * bond + g;
        let r_idx = g * 2u * dr + t * dr + r;
        sum = cmplx_add(sum, cmplx_mul(left_buf[l_idx], right_buf[r_idx]));
    }
    theta_buf[idx] = sum;
}
