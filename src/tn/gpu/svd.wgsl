// Double-float complex kernels for resident MPS splits.
// vec2f = (hi, lo) real. vec4f = (re_hi, re_lo, im_hi, im_lo).

fn two_sum(a: f32, b: f32) -> vec2f {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return vec2f(s, e);
}

fn split_f32(a: f32) -> vec2f {
    let t = a * 4097.0;
    let hi = t - (t - a);
    return vec2f(hi, a - hi);
}

fn two_prod(a: f32, b: f32) -> vec2f {
    let p = a * b;
    let aa = split_f32(a);
    let bb = split_f32(b);
    let err = ((aa.x * bb.x - p) + aa.x * bb.y + aa.y * bb.x) + aa.y * bb.y;
    return vec2f(p, err);
}

fn df_renorm(a: vec2f) -> vec2f {
    return two_sum(a.x, a.y);
}

fn df_add(a: vec2f, b: vec2f) -> vec2f {
    let s = two_sum(a.x, b.x);
    return df_renorm(vec2f(s.x, s.y + a.y + b.y));
}

fn df_sub(a: vec2f, b: vec2f) -> vec2f {
    return df_add(a, vec2f(-b.x, -b.y));
}

fn df_mul(a: vec2f, b: vec2f) -> vec2f {
    let p = two_prod(a.x, b.x);
    return df_renorm(vec2f(p.x, p.y + a.x * b.y + a.y * b.x));
}

fn df_div(a: vec2f, b: vec2f) -> vec2f {
    let q = a.x / b.x;
    let t = two_prod(q, b.x);
    let r = df_sub(a, vec2f(t.x, t.y));
    let q2 = r.x / b.x;
    return df_add(vec2f(q, 0.0), vec2f(q2, 0.0));
}

fn df_abs(a: vec2f) -> vec2f {
    if (a.x < 0.0) {
        return vec2f(-a.x, -a.y);
    }
    return a;
}

fn df_sqrt(a: vec2f) -> vec2f {
    if (a.x <= 0.0) {
        return vec2f(0.0, 0.0);
    }
    let r = sqrt(a.x);
    let r2 = df_mul(vec2f(r, 0.0), vec2f(r, 0.0));
    let corr = df_div(df_sub(a, r2), vec2f(2.0 * r, 0.0));
    return df_add(vec2f(r, 0.0), corr);
}

fn cf_add(a: vec4f, b: vec4f) -> vec4f {
    let re = df_add(a.xy, b.xy);
    let im = df_add(a.zw, b.zw);
    return vec4f(re, im);
}

fn cf_sub(a: vec4f, b: vec4f) -> vec4f {
    return cf_add(a, vec4f(-b.x, -b.y, -b.z, -b.w));
}

fn cf_mul(a: vec4f, b: vec4f) -> vec4f {
    let arbr = df_mul(a.xy, b.xy);
    let aibi = df_mul(a.zw, b.zw);
    let arbi = df_mul(a.xy, b.zw);
    let aibr = df_mul(a.zw, b.xy);
    let re = df_sub(arbr, aibi);
    let im = df_add(arbi, aibr);
    return vec4f(re, im);
}

fn cf_conj(a: vec4f) -> vec4f {
    return vec4f(a.x, a.y, -a.z, -a.w);
}

fn cf_scale(a: vec4f, s: vec2f) -> vec4f {
    return vec4f(df_mul(a.xy, s), df_mul(a.zw, s));
}

fn cf_norm_sqr(a: vec4f) -> vec2f {
    return df_add(df_mul(a.xy, a.xy), df_mul(a.zw, a.zw));
}

fn cf_abs(a: vec4f) -> vec2f {
    return df_sqrt(cf_norm_sqr(a));
}

// KERNEL apply_one_qubit_df
struct OneQubitDfParams {
    left: u32,
    right: u32,
    _pad0: u32,
    _pad1: u32,
    u00: vec4f,
    u01: vec4f,
    u10: vec4f,
    u11: vec4f,
}

@group(0) @binding(0) var<uniform> one_df_params: OneQubitDfParams;
@group(0) @binding(1) var<storage, read_write> one_df_site: array<vec4f>;

@compute @workgroup_size(64)
fn apply_one_qubit_df(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pairs = one_df_params.left * one_df_params.right;
    let idx = gid.x;
    if (idx >= pairs) {
        return;
    }
    let b = idx % one_df_params.right;
    let a = idx / one_df_params.right;
    let stride = one_df_params.right;
    let base = a * 2u * stride + b;
    let v0 = one_df_site[base];
    let v1 = one_df_site[base + stride];
    one_df_site[base] = cf_add(cf_mul(one_df_params.u00, v0), cf_mul(one_df_params.u01, v1));
    one_df_site[base + stride] = cf_add(cf_mul(one_df_params.u10, v0), cf_mul(one_df_params.u11, v1));
}

// KERNEL merge_two_site_df
struct MergeDfParams {
    dl: u32,
    dr: u32,
    bond: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> merge_df_params: MergeDfParams;
@group(0) @binding(1) var<storage, read> merge_df_left: array<vec4f>;
@group(0) @binding(2) var<storage, read> merge_df_right: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> merge_df_theta: array<vec4f>;

@compute @workgroup_size(64)
fn merge_two_site_df(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dl = merge_df_params.dl;
    let dr = merge_df_params.dr;
    let bond = merge_df_params.bond;
    let total = dl * 4u * dr;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let r = idx % dr;
    let t = (idx / dr) % 2u;
    let s = (idx / (dr * 2u)) % 2u;
    let a = idx / (dr * 4u);
    var sum = vec4f(0.0, 0.0, 0.0, 0.0);
    for (var g: u32 = 0u; g < bond; g = g + 1u) {
        let l_idx = a * 2u * bond + s * bond + g;
        let r_idx = g * 2u * dr + t * dr + r;
        sum = cf_add(sum, cf_mul(merge_df_left[l_idx], merge_df_right[r_idx]));
    }
    merge_df_theta[idx] = sum;
}

// KERNEL apply_two_qubit_df
struct UnitaryDfParams {
    dl: u32,
    dr: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> unit_df_params: UnitaryDfParams;
@group(0) @binding(1) var<storage, read> unit_df_u: array<vec4f>;
@group(0) @binding(2) var<storage, read> unit_df_in: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> unit_df_out: array<vec4f>;

@compute @workgroup_size(64)
fn apply_two_qubit_df(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dr = unit_df_params.dr;
    let total = unit_df_params.dl * 4u * dr;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let r = idx % dr;
    let t_out = (idx / dr) % 2u;
    let s_out = (idx / (dr * 2u)) % 2u;
    let a = idx / (dr * 4u);
    var sum = vec4f(0.0, 0.0, 0.0, 0.0);
    for (var s_in: u32 = 0u; s_in < 2u; s_in = s_in + 1u) {
        for (var t_in: u32 = 0u; t_in < 2u; t_in = t_in + 1u) {
            let u_idx = ((s_out * 2u + t_out) * 2u + s_in) * 2u + t_in;
            let in_idx = ((a * 2u + s_in) * 2u + t_in) * dr + r;
            sum = cf_add(sum, cf_mul(unit_df_u[u_idx], unit_df_in[in_idx]));
        }
    }
    unit_df_out[idx] = sum;
}

// KERNEL jacobi_all
struct JacobiParams {
    rows: u32,
    cols: u32,
    sweeps: u32,
    steps: u32,
    pairs_per_step: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> jacobi_params: JacobiParams;
@group(0) @binding(1) var<storage, read> jacobi_pairs: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> jacobi_a: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> jacobi_v: array<vec4f>;

// One workgroup owns every pair of a step (pairs_per_step ≤ 128). Steps and
// sweeps stay inside this dispatch so the host does not wait between them.
@compute @workgroup_size(256)
fn jacobi_all(@builtin(local_invocation_id) lid: vec3<u32>) {
    let pair = lid.x;
    let rows = jacobi_params.rows;
    let cols = jacobi_params.cols;
    let pairs_per_step = jacobi_params.pairs_per_step;
    for (var sweep: u32 = 0u; sweep < jacobi_params.sweeps; sweep = sweep + 1u) {
        for (var step: u32 = 0u; step < jacobi_params.steps; step = step + 1u) {
            if (pair < pairs_per_step) {
                let pair_index = step * pairs_per_step + pair;
                let p = jacobi_pairs[pair_index].x;
                let q = jacobi_pairs[pair_index].y;

                var app = vec2f(0.0, 0.0);
                var aqq = vec2f(0.0, 0.0);
                var apq = vec4f(0.0, 0.0, 0.0, 0.0);
                for (var i: u32 = 0u; i < rows; i = i + 1u) {
                    let xp = jacobi_a[i * cols + p];
                    let xq = jacobi_a[i * cols + q];
                    app = df_add(app, cf_norm_sqr(xp));
                    aqq = df_add(aqq, cf_norm_sqr(xq));
                    apq = cf_add(apq, cf_mul(cf_conj(xp), xq));
                }
                let g = cf_abs(apq);
                let scale = sqrt(max(app.x, 0.0) * max(aqq.x, 0.0));
                if (g.x > 1e-12 * scale) {
                    let rot = cf_scale(cf_conj(apq), df_div(vec2f(1.0, 0.0), g));
                    let tau = df_div(df_sub(aqq, app), df_mul(vec2f(2.0, 0.0), g));
                    let tau_abs = df_abs(tau);
                    let inner = df_sqrt(df_add(vec2f(1.0, 0.0), df_mul(tau, tau)));
                    let t_abs = df_div(vec2f(1.0, 0.0), df_add(tau_abs, inner));
                    var t = t_abs;
                    if (tau.x < 0.0) {
                        t = vec2f(-t_abs.x, -t_abs.y);
                    }
                    let c = df_div(vec2f(1.0, 0.0), df_sqrt(df_add(vec2f(1.0, 0.0), df_mul(t, t))));
                    let s = df_mul(c, t);

                    for (var i: u32 = 0u; i < rows; i = i + 1u) {
                        let idx_p = i * cols + p;
                        let idx_q = i * cols + q;
                        let xp = jacobi_a[idx_p];
                        let xq = cf_mul(jacobi_a[idx_q], rot);
                        jacobi_a[idx_p] = cf_sub(cf_scale(xp, c), cf_scale(xq, s));
                        jacobi_a[idx_q] = cf_add(cf_scale(xp, s), cf_scale(xq, c));
                    }
                    for (var i: u32 = 0u; i < cols; i = i + 1u) {
                        let idx_p = i * cols + p;
                        let idx_q = i * cols + q;
                        let xp = jacobi_v[idx_p];
                        let xq = cf_mul(jacobi_v[idx_q], rot);
                        jacobi_v[idx_p] = cf_sub(cf_scale(xp, c), cf_scale(xq, s));
                        jacobi_v[idx_q] = cf_add(cf_scale(xp, s), cf_scale(xq, c));
                    }
                }
            }
            storageBarrier();
            workgroupBarrier();
        }
    }
}

// KERNEL column_norms
struct NormParams {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> norm_params: NormParams;
@group(0) @binding(1) var<storage, read> norm_a: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> norm_out: array<vec2f>;

@compute @workgroup_size(64)
fn column_norms(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= norm_params.cols) {
        return;
    }
    var nrm = vec2f(0.0, 0.0);
    for (var i: u32 = 0u; i < norm_params.rows; i = i + 1u) {
        nrm = df_add(nrm, cf_norm_sqr(norm_a[i * norm_params.cols + col]));
    }
    norm_out[col] = df_sqrt(nrm);
}
