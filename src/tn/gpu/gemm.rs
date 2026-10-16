//! Tiled complex f32 products, and the side threshold that chooses them.
//!
//! The side is the square matrix with the same multiply count, not the qubit
//! count. A 2-site merge is `(2·dl) × bond × (2·dr)`.

use ndarray::{Array3, Array4};
use num_complex::Complex64;
use wgpu::util::DeviceExt;

use super::device::GpuMpsDevice;

/// Side used before this process has measured the machine, and when measurement
/// has not run (tests, CPU backend). This Mac's upload/multiply/readback
/// crossover was 160 (2026-10-16).
const GPU_GEMM_MIN_SIDE_FALLBACK: usize = 160;

/// Bump when the kernel or the timing method changes, so old cache files are
/// measured again.
const GEMM_CALIBRATION_VERSION: u32 = 1;

static GEMM_MIN_SIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(GPU_GEMM_MIN_SIDE_FALLBACK);

pub(crate) fn gemm_min_side() -> usize {
    GEMM_MIN_SIDE.load(std::sync::atomic::Ordering::Relaxed)
}

fn set_gemm_min_side(side: usize) {
    GEMM_MIN_SIDE.store(side, std::sync::atomic::Ordering::Relaxed);
}

/// True when an m×k by k×n product has at least as many multiplies as a square
/// of side `min_side`.
fn merge_uses_gpu_at(m: usize, k: usize, n: usize, min_side: usize) -> bool {
    if m == 0 || k == 0 || n == 0 || min_side == 0 {
        return false;
    }
    let flops = (m as u128)
        .saturating_mul(k as u128)
        .saturating_mul(n as u128);
    let side = min_side as u128;
    flops >= side.saturating_mul(side).saturating_mul(side)
}

/// True when this process's measured side says the product should run on the GPU.
pub(crate) fn use_gpu_gemm(m: usize, k: usize, n: usize) -> bool {
    use_gpu_gemm_batch(1, m, k, n)
}

/// Same rule as [`use_gpu_gemm`], counting every item of a batched product.
pub(crate) fn use_gpu_gemm_batch(batch: usize, m: usize, k: usize, n: usize) -> bool {
    if batch == 0 {
        return false;
    }
    let flops = (batch as u128)
        .saturating_mul(m as u128)
        .saturating_mul(k as u128)
        .saturating_mul(n as u128);
    let side = gemm_min_side() as u128;
    m != 0 && k != 0 && n != 0 && flops >= side.saturating_mul(side).saturating_mul(side)
}

/// Smallest measured side where the GPU round trip was faster, if any sample won.
fn smallest_gpu_win(samples: &[(usize, u128, u128)]) -> Option<usize> {
    samples
        .iter()
        .filter(|(_, cpu_us, gpu_us)| *gpu_us < *cpu_us)
        .map(|(side, _, _)| *side)
        .min()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmParams {
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
}

pub(super) struct GemmKernels {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl GemmKernels {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm-f32"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gemm.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm-f32"),
            entries: &[
                uniform(0),
                storage(1, true),
                storage(2, true),
                storage(3, false),
            ],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemm_f32"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("gemm_f32"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &module,
            entry_point: Some("gemm_f32"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }
}

fn fill_pair(n: usize) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let scale = 1.0 / (n as f32).sqrt();
    let mut a = Vec::with_capacity(n * n);
    let mut b = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            a.push([
                scale * ((i * 3 + j) % 17) as f32 / 17.0,
                scale * ((i + j * 5) % 13) as f32 / 13.0,
            ]);
            b.push([
                scale * ((i * 7 + j) % 19) as f32 / 19.0,
                scale * ((i * 2 + j * 3) % 11) as f32 / 11.0,
            ]);
        }
    }
    (a, b)
}

fn cpu_matrices_f64(
    n: usize,
    a: &[[f32; 2]],
    b: &[[f32; 2]],
) -> (
    nalgebra::DMatrix<num_complex::Complex64>,
    nalgebra::DMatrix<num_complex::Complex64>,
) {
    use nalgebra::DMatrix;
    use num_complex::Complex64;
    let mut ma = DMatrix::<Complex64>::zeros(n, n);
    let mut mb = DMatrix::<Complex64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let pa = a[i * n + j];
            let pb = b[i * n + j];
            ma[(i, j)] = Complex64::new(f64::from(pa[0]), f64::from(pa[1]));
            mb[(i, j)] = Complex64::new(f64::from(pb[0]), f64::from(pb[1]));
        }
    }
    (ma, mb)
}

fn cpu_matrices_f32(
    n: usize,
    a: &[[f32; 2]],
    b: &[[f32; 2]],
) -> (
    nalgebra::DMatrix<num_complex::Complex32>,
    nalgebra::DMatrix<num_complex::Complex32>,
) {
    use nalgebra::DMatrix;
    use num_complex::Complex32;
    let mut ma = DMatrix::<Complex32>::zeros(n, n);
    let mut mb = DMatrix::<Complex32>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let pa = a[i * n + j];
            let pb = b[i * n + j];
            ma[(i, j)] = Complex32::new(pa[0], pa[1]);
            mb[(i, j)] = Complex32::new(pb[0], pb[1]);
        }
    }
    (ma, mb)
}

struct GpuGemm {
    device: std::sync::Arc<GpuMpsDevice>,
    kernels: GemmKernels,
}

impl GpuGemm {
    fn new() -> Option<Self> {
        let device = GpuMpsDevice::try_new()?;
        let kernels = GemmKernels::new(device.device());
        Some(Self { device, kernels })
    }

    /// Upload A and B once, then time only the product. C stays on the device.
    fn prepare(&self, n: usize, a: &[[f32; 2]], b: &[[f32; 2]]) -> ResidentGemm<'_> {
        self.prepare_rect(n, n, n, a, b)
    }

    fn prepare_rect(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[[f32; 2]],
        b: &[[f32; 2]],
    ) -> ResidentGemm<'_> {
        let (c_buf, keep) = upload_gemm(&self.device, &self.kernels, (1, m, k, n), a, b);
        ResidentGemm {
            gpu: self,
            m,
            n,
            c_buf,
            _keep: keep,
        }
    }
}

struct ResidentGemm<'a> {
    gpu: &'a GpuGemm,
    m: usize,
    n: usize,
    c_buf: wgpu::Buffer,
    _keep: (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup),
}

impl ResidentGemm<'_> {
    fn multiply(&self) {
        let groups_x = (self.n as u32).div_ceil(16);
        let groups_y = (self.m as u32).div_ceil(16);
        let mut encoder =
            self.gpu
                .device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gemm"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gpu.kernels.pipeline);
            pass.set_bind_group(0, &self._keep.3, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.gpu.device.queue().submit(Some(encoder.finish()));
        self.gpu.device.device().poll(wgpu::Maintain::Wait);
    }
}

fn upload_gemm(
    gpu: &GpuMpsDevice,
    kernels: &GemmKernels,
    dims: (usize, usize, usize, usize),
    a: &[[f32; 2]],
    b: &[[f32; 2]],
) -> (
    wgpu::Buffer,
    (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup),
) {
    let (batch, m, k, n) = dims;
    let a_buf = storage_buf(gpu, "gemm-a", bytemuck::cast_slice(a));
    let b_buf = storage_buf(gpu, "gemm-b", bytemuck::cast_slice(b));
    let c_buf = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm-c"),
        size: (batch.max(1) * m * n * std::mem::size_of::<[f32; 2]>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = GemmParams {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        batch: batch.max(1) as u32,
    };
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gemm-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gemm-bind"),
        layout: &kernels.layout,
        entries: &[
            entry(0, &params_buf),
            entry(1, &a_buf),
            entry(2, &b_buf),
            entry(3, &c_buf),
        ],
    });
    gpu.track_allocation(a_buf.size() + b_buf.size() + c_buf.size() + params_buf.size());
    (c_buf, (a_buf, b_buf, params_buf, bind))
}

impl GpuMpsDevice {
    /// Merge two site tensors on the GPU when the product is at least this
    /// machine's measured side. `None` means the caller should use CPU f64.
    pub(crate) fn try_gpu_merge(
        &self,
        left: &Array3<Complex64>,
        right: &Array3<Complex64>,
    ) -> Option<Array4<Complex64>> {
        let dl = left.dim().0;
        let dr = right.dim().2;
        let bond = left.dim().2.min(right.dim().0);
        let m = dl * 2;
        let n = 2 * dr;
        if !use_gpu_gemm(m, bond, n) {
            return None;
        }
        let mut a = vec![[0.0f32; 2]; m * bond];
        let mut b = vec![[0.0f32; 2]; bond * n];
        for a_i in 0..dl {
            for s in 0..2 {
                for g in 0..bond {
                    let z = left[[a_i, s, g]];
                    a[(a_i * 2 + s) * bond + g] = [z.re as f32, z.im as f32];
                }
            }
        }
        for g in 0..bond {
            for t in 0..2 {
                for b_i in 0..dr {
                    let z = right[[g, t, b_i]];
                    b[g * n + t * dr + b_i] = [z.re as f32, z.im as f32];
                }
            }
        }
        let kernels = self.gemm_kernels();
        let (c_buf, keep) = upload_gemm(self, kernels, (1, m, bond, n), &a, &b);
        let groups_x = (n as u32).div_ceil(16);
        let groups_y = (m as u32).div_ceil(16);
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemm-merge"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm-merge"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernels.pipeline);
            pass.set_bind_group(0, &keep.3, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue().submit(Some(encoder.finish()));
        self.device().poll(wgpu::Maintain::Wait);
        let flat = read_complex(self, &c_buf, m * n);
        drop(keep);
        let mut theta = Array4::<Complex64>::zeros((dl, 2, 2, dr));
        for a_i in 0..dl {
            for s in 0..2 {
                for t in 0..2 {
                    for b_i in 0..dr {
                        let g = flat[(a_i * 2 + s) * n + t * dr + b_i];
                        theta[[a_i, s, t, b_i]] = Complex64::new(f64::from(g[0]), f64::from(g[1]));
                    }
                }
            }
        }
        Some(theta)
    }

    /// One dispatch of `batch` independent products. `None` when this machine's
    /// side says the CPU product is faster. `A` stays on the device for that
    /// single multiply; the caller keeps the result.
    pub(crate) fn try_complex_gemm(
        &self,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        a: &[Complex64],
        b: &[Complex64],
    ) -> Option<Vec<Complex64>> {
        if !use_gpu_gemm_batch(batch, m, k, n) {
            return None;
        }
        if a.len() != batch * m * k || b.len() != batch * k * n {
            return None;
        }
        let af: Vec<[f32; 2]> = a.iter().map(|z| [z.re as f32, z.im as f32]).collect();
        let bf: Vec<[f32; 2]> = b.iter().map(|z| [z.re as f32, z.im as f32]).collect();
        let kernels = self.gemm_kernels();
        let (c_buf, keep) = upload_gemm(self, kernels, (batch, m, k, n), &af, &bf);
        dispatch_uploaded(self, kernels, m, n, batch, &keep.3);
        let flat = read_complex(self, &c_buf, batch * m * n);
        drop(keep);
        Some(
            flat.into_iter()
                .map(|z| Complex64::new(f64::from(z[0]), f64::from(z[1])))
                .collect(),
        )
    }
}

fn read_complex(gpu: &GpuMpsDevice, buf: &wgpu::Buffer, count: usize) -> Vec<[f32; 2]> {
    let bytes = (count * std::mem::size_of::<[f32; 2]>()) as u64;
    let readback = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm-read-all"),
        size: bytes.max(8),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemm-read-all"),
        });
    encoder.copy_buffer_to_buffer(buf, 0, &readback, 0, bytes);
    gpu.queue().submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device().poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback.unmap();
    out
}

fn storage_buf(gpu: &GpuMpsDevice, label: &str, contents: &[u8]) -> wgpu::Buffer {
    gpu.device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
}

impl GpuGemm {
    fn read_corner(&self, buf: &wgpu::Buffer) -> [f32; 2] {
        let bytes = std::mem::size_of::<[f32; 2]>() as u64;
        let readback = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemm-read"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gemm-read"),
                });
        encoder.copy_buffer_to_buffer(buf, 0, &readback, 0, bytes);
        self.device.queue().submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.device().poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let corner: [f32; 2] = *bytemuck::from_bytes(&data[..8]);
        drop(data);
        readback.unmap();
        corner
    }

    fn read_all(&self, buf: &wgpu::Buffer, n: usize) -> Vec<[f32; 2]> {
        let bytes = (n * n * std::mem::size_of::<[f32; 2]>()) as u64;
        let readback = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemm-read-all"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gemm-read-all"),
                });
        encoder.copy_buffer_to_buffer(buf, 0, &readback, 0, bytes);
        self.device.queue().submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.device().poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let out = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();
        out
    }
}

/// Measure this machine once, then reuse the side on later starts.
///
/// Compares a square complex product: CPU f64 versus GPU f32 including the
/// upload and the readback the merge pays before the CPU SVD. Stops within
/// about two minutes. `WQC_GEMM_MIN_SIDE` skips the measurement.
pub(crate) fn calibrate_gemm_side(gpu: &GpuMpsDevice) {
    if let Some(side) = std::env::var("WQC_GEMM_MIN_SIDE")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|side| *side >= 16)
    {
        set_gemm_min_side(side);
        println!("wqc-core: GPU merge from side {side} (WQC_GEMM_MIN_SIDE)");
        return;
    }

    let fingerprint = gemm_fingerprint(gpu.adapter_label());
    let path = gemm_side_cache_path();
    if let Some(side) = read_side_cache(&path, &fingerprint) {
        set_gemm_min_side(side);
        announce_gemm_side(side, "cached");
        return;
    }

    println!(
        "wqc-core: measuring GPU merge side for this machine (first webgpu start, up to 2 min)"
    );
    let _ = gpu.gemm_kernels();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(110);
    let coarse = [64usize, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048];
    let mut samples = Vec::new();
    for n in coarse {
        if std::time::Instant::now() >= deadline {
            break;
        }
        let (cpu_us, gpu_us) = measure_square(gpu, n);
        println!("wqc-core: side {n} cpu {cpu_us} µs  gpu {gpu_us} µs");
        let gpu_faster = gpu_us < cpu_us;
        samples.push((n, cpu_us, gpu_us));
        if gpu_faster {
            refine_side(gpu, &mut samples, deadline);
            break;
        }
    }

    let side = smallest_gpu_win(&samples).unwrap_or(usize::MAX / 4);
    set_gemm_min_side(side);
    let how = if write_side_cache(&path, &fingerprint, side) {
        format!("saved {}", path.display())
    } else {
        "not saved; cache path unavailable".to_string()
    };
    announce_gemm_side(side, &how);
}

fn announce_gemm_side(side: usize, how: &str) {
    if side > 4096 {
        println!("wqc-core: GPU merge stayed slower on this machine; merges stay on CPU ({how})");
    } else {
        println!("wqc-core: GPU merge from side {side} ({how})");
    }
}

fn refine_side(
    gpu: &GpuMpsDevice,
    samples: &mut Vec<(usize, u128, u128)>,
    deadline: std::time::Instant,
) {
    let Some(mut hi) = smallest_gpu_win(samples) else {
        return;
    };
    let Some(mut lo) = samples
        .iter()
        .filter(|(side, cpu_us, gpu_us)| *side < hi && gpu_us >= cpu_us)
        .map(|(side, _, _)| *side)
        .max()
    else {
        return;
    };
    while hi - lo > 16 && std::time::Instant::now() < deadline {
        let mid = ((lo + hi) / 2) / 16 * 16;
        if mid <= lo || mid >= hi {
            break;
        }
        let (cpu_us, gpu_us) = measure_square(gpu, mid);
        println!("wqc-core: side {mid} cpu {cpu_us} µs  gpu {gpu_us} µs");
        samples.push((mid, cpu_us, gpu_us));
        if gpu_us < cpu_us {
            hi = mid;
        } else {
            lo = mid;
        }
    }
}

fn measure_square(gpu: &GpuMpsDevice, n: usize) -> (u128, u128) {
    let (a, b) = fill_pair(n);
    let (ma, mb) = cpu_matrices_f64(n, &a, &b);
    let reps = if n >= 1024 { 1 } else { 3 };
    let _ = &ma * &mb;
    let mut cpu = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = std::time::Instant::now();
        let c = &ma * &mb;
        std::hint::black_box(c[(0, 0)]);
        cpu.push(start.elapsed().as_micros());
    }
    let kernels = gpu.gemm_kernels();
    let warmup = upload_gemm(gpu, kernels, (1, n, n, n), &a, &b);
    dispatch_uploaded(gpu, kernels, n, n, 1, &warmup.1 .3);
    let _ = read_complex(gpu, &warmup.0, n * n);
    drop(warmup);
    let mut gpu_us = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = std::time::Instant::now();
        let uploaded = upload_gemm(gpu, kernels, (1, n, n, n), &a, &b);
        dispatch_uploaded(gpu, kernels, n, n, 1, &uploaded.1 .3);
        let corner = read_complex(gpu, &uploaded.0, n * n);
        std::hint::black_box(corner.first().copied());
        drop(uploaded);
        gpu_us.push(start.elapsed().as_micros());
    }
    (median_us(&mut cpu), median_us(&mut gpu_us))
}

fn dispatch_uploaded(
    gpu: &GpuMpsDevice,
    kernels: &GemmKernels,
    m: usize,
    n: usize,
    batch: usize,
    bind: &wgpu::BindGroup,
) {
    let groups_x = (n as u32).div_ceil(16);
    let groups_y = (m as u32).div_ceil(16);
    let groups_z = batch.max(1) as u32;
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemm"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernels.pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, groups_z);
    }
    gpu.queue().submit(Some(encoder.finish()));
    gpu.device().poll(wgpu::Maintain::Wait);
}

fn median_us(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn gemm_fingerprint(adapter: &str) -> String {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let rayon = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());
    let adapter = adapter.replace(['\n', '\r', '|'], " ");
    format!("v{GEMM_CALIBRATION_VERSION}|{adapter}|threads={threads}|rayon={rayon}")
}

fn gemm_side_cache_path() -> std::path::PathBuf {
    if let Some(path) = std::env::var_os("WQC_GEMM_SIDE_PATH") {
        return std::path::PathBuf::from(path);
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            if cfg!(windows) {
                std::env::var_os("LOCALAPPDATA").map(std::path::PathBuf::from)
            } else if cfg!(target_os = "macos") {
                std::env::var_os("HOME")
                    .map(|home| std::path::PathBuf::from(home).join("Library/Caches"))
            } else {
                std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".cache"))
            }
        })
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    base.join("wqc").join("gemm-min-side")
}

fn read_side_cache(path: &std::path::Path, fingerprint: &str) -> Option<usize> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    let stored_fp = lines.next()?;
    let side = lines.next()?.parse().ok()?;
    if stored_fp == fingerprint && side >= 16 {
        Some(side)
    } else {
        None
    }
}

fn write_side_cache(path: &std::path::Path, fingerprint: &str, side: usize) -> bool {
    if let Some(dir) = path.parent() {
        if std::fs::create_dir_all(dir).is_err() {
            return false;
        }
    }
    let tmp = path.with_extension("tmp");
    let body = format!("{fingerprint}\n{side}\n");
    if std::fs::write(&tmp, body).is_err() {
        return false;
    }
    std::fs::rename(&tmp, path).is_ok()
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn time_ms(mut body: impl FnMut()) -> u128 {
        let start = std::time::Instant::now();
        body();
        start.elapsed().as_millis()
    }

    fn time_us(mut body: impl FnMut()) -> u128 {
        let start = std::time::Instant::now();
        body();
        start.elapsed().as_micros()
    }

    #[test]
    fn gemm_switch_follows_multiply_count() {
        let side = GPU_GEMM_MIN_SIDE_FALLBACK;
        assert!(
            !merge_uses_gpu_at(64, 64, 64, side),
            "side 64 was slower on the GPU"
        );
        assert!(merge_uses_gpu_at(side, side, side, side));
        // Saturated χ=128 merge: (2·128) × 128 × (2·128), about side 203.
        assert!(merge_uses_gpu_at(256, 128, 256, side));
        // A long side with almost no shared dimension is still a small product.
        assert!(!merge_uses_gpu_at(512, 2, 512, side));
        assert!(!merge_uses_gpu_at(0, 128, 128, side));
        assert_eq!(
            smallest_gpu_win(&[(64, 100, 400), (128, 2000, 1400), (192, 7000, 3000)]),
            Some(128)
        );
        assert_eq!(smallest_gpu_win(&[(256, 16_000, 20_000)]), None);
    }

    #[test]
    fn gemm_side_cache_rejects_a_different_machine() {
        let path = std::env::temp_dir().join(format!("wqc-gemm-side-{}-cache", std::process::id()));
        let fingerprint = "v1|test-gpu|threads=1|rayon=1";
        assert!(write_side_cache(&path, fingerprint, 192));
        assert_eq!(read_side_cache(&path, fingerprint), Some(192));
        assert_eq!(
            read_side_cache(&path, "v1|other-gpu|threads=1|rayon=1"),
            None
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[ignore = "measures this machine's GPU merge side; up to 2 min"]
    fn calibrate_gemm_side_on_this_machine() {
        let Some(gpu) = GpuMpsDevice::try_new() else {
            eprintln!("skip calibrate_gemm_side_on_this_machine: no WebGPU adapter");
            return;
        };
        let path = std::env::temp_dir().join(format!("wqc-gemm-cal-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        std::env::set_var("WQC_GEMM_SIDE_PATH", &path);
        std::env::remove_var("WQC_GEMM_MIN_SIDE");
        let started = std::time::Instant::now();
        calibrate_gemm_side(&gpu);
        let first_ms = started.elapsed().as_millis();
        let side = gemm_min_side();
        let started = std::time::Instant::now();
        calibrate_gemm_side(&gpu);
        let second_ms = started.elapsed().as_millis();
        eprintln!("calibrated side={side} first_ms={first_ms} second_ms={second_ms}");
        assert!(first_ms < 120_000, "calibration took {first_ms} ms");
        assert!(second_ms < 1_000, "cached load took {second_ms} ms");
        if side <= 4096 {
            assert!(side >= 64 && side.is_multiple_of(16), "side {side}");
        }
        assert!(path.is_file());
        std::env::remove_var("WQC_GEMM_SIDE_PATH");
        set_gemm_min_side(GPU_GEMM_MIN_SIDE_FALLBACK);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gpu_merge_above_side_matches_f64() {
        set_gemm_min_side(GPU_GEMM_MIN_SIDE_FALLBACK);
        let Some(gpu) = GpuMpsDevice::try_new() else {
            eprintln!("skip gpu_merge_above_side_matches_f64: no WebGPU adapter");
            return;
        };
        let dl = 88usize;
        let bond = 144usize;
        let dr = 84usize;
        let mut left = Array3::<Complex64>::zeros((dl, 2, bond));
        let mut right = Array3::<Complex64>::zeros((bond, 2, dr));
        let scale = 1.0 / (bond as f64).sqrt();
        for a in 0..dl {
            for s in 0..2 {
                for g in 0..bond {
                    left[[a, s, g]] = Complex64::new(
                        scale * ((a * 3 + s + g) % 17) as f64 / 17.0,
                        scale * ((a + s * 5 + g) % 13) as f64 / 13.0,
                    );
                }
            }
        }
        for g in 0..bond {
            for t in 0..2 {
                for b in 0..dr {
                    right[[g, t, b]] = Complex64::new(
                        scale * ((g * 7 + t + b) % 19) as f64 / 19.0,
                        scale * ((g * 2 + t * 3 + b) % 11) as f64 / 11.0,
                    );
                }
            }
        }
        let mut cpu = Array4::<Complex64>::zeros((dl, 2, 2, dr));
        for a in 0..dl {
            for b in 0..dr {
                for s in 0..2 {
                    for t in 0..2 {
                        let mut sum = Complex64::new(0.0, 0.0);
                        for g in 0..bond {
                            sum += left[[a, s, g]] * right[[g, t, b]];
                        }
                        cpu[[a, s, t, b]] = sum;
                    }
                }
            }
        }
        let got = gpu
            .try_gpu_merge(&left, &right)
            .expect("side is above the threshold");
        let mut max_err = 0.0f64;
        for a in 0..dl {
            for s in 0..2 {
                for t in 0..2 {
                    for b in 0..dr {
                        let d = got[[a, s, t, b]] - cpu[[a, s, t, b]];
                        max_err = max_err.max(d.re.abs()).max(d.im.abs());
                    }
                }
            }
        }
        assert!(max_err < 1e-5, "gpu merge max abs err {max_err}");
        let tiny_l = Array3::<Complex64>::zeros((1, 2, 1));
        let tiny_r = Array3::<Complex64>::zeros((1, 2, 1));
        assert!(gpu.try_gpu_merge(&tiny_l, &tiny_r).is_none());
    }

    #[test]
    #[ignore = "large complex GEMM, CPU f64 / CPU f32 / resident GPU f32"]
    fn gemm_size_crossover() {
        let Some(gpu) = GpuGemm::new() else {
            eprintln!("skip gemm_size_crossover: no WebGPU adapter");
            return;
        };
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        eprintln!("host_threads={threads}");

        let n = 64usize;
        let (a, b) = fill_pair(n);
        let (ma, mb) = cpu_matrices_f32(n, &a, &b);
        let cpu = &ma * &mb;
        let resident = gpu.prepare(n, &a, &b);
        resident.multiply();
        let gpu_corner = gpu.read_corner(&resident.c_buf);
        let err = (cpu[(0, 0)].re - gpu_corner[0])
            .abs()
            .max((cpu[(0, 0)].im - gpu_corner[1]).abs());
        assert!(
            err < 1e-3,
            "n=64 corner cpu {}+{}i gpu {:?}",
            cpu[(0, 0)].re,
            cpu[(0, 0)].im,
            gpu_corner
        );
        for n in [64usize, 128] {
            let (a, b) = fill_pair(n);
            let (ma, mb) = cpu_matrices_f64(n, &a, &b);
            let cpu = &ma * &mb;
            let resident = gpu.prepare(n, &a, &b);
            resident.multiply();
            let gpu_c = gpu.read_all(&resident.c_buf, n);
            let mut max_err = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    let g = gpu_c[i * n + j];
                    let c = cpu[(i, j)];
                    max_err = max_err.max((c.re - f64::from(g[0])).abs());
                    max_err = max_err.max((c.im - f64::from(g[1])).abs());
                }
            }
            eprintln!("max_abs_err n={n} {max_err}");
        }

        for n in [16usize, 32, 48, 64, 96, 128, 256] {
            let (a, b) = fill_pair(n);
            let (ma64, mb64) = cpu_matrices_f64(n, &a, &b);
            let _ = &ma64 * &mb64;
            let cpu64 = time_us(|| {
                let c = &ma64 * &mb64;
                std::hint::black_box(c[(0, 0)]);
            });
            let resident = gpu.prepare(n, &a, &b);
            resident.multiply();
            let gpu32 = time_us(|| resident.multiply());
            eprintln!("boundary n={n} cpu_f64_us={cpu64} gpu_f32_resident_us={gpu32}");
        }

        for n in [128usize, 160, 192, 208, 224, 240, 256] {
            let (a, b) = fill_pair(n);
            let (ma64, mb64) = cpu_matrices_f64(n, &a, &b);
            let _ = &ma64 * &mb64;
            let cpu64 = time_us(|| {
                let c = &ma64 * &mb64;
                std::hint::black_box(c[(0, 0)]);
            });
            let round = time_us(|| {
                let resident = gpu.prepare(n, &a, &b);
                resident.multiply();
                let c = gpu.read_all(&resident.c_buf, n);
                std::hint::black_box(c[0]);
            });
            eprintln!("roundtrip n={n} cpu_f64_us={cpu64} gpu_f32_roundtrip_us={round}");
        }

        if std::env::var_os("WQC_GEMM_LARGE").is_some() {
            for n in [512usize, 1024, 2048, 4096] {
                let (a, b) = fill_pair(n);
                let (ma64, mb64) = cpu_matrices_f64(n, &a, &b);
                let _ = &ma64 * &mb64;
                let cpu64 = time_ms(|| {
                    let c = &ma64 * &mb64;
                    std::hint::black_box(c[(0, 0)]);
                });
                let (ma32, mb32) = cpu_matrices_f32(n, &a, &b);
                let _ = &ma32 * &mb32;
                let cpu32 = time_ms(|| {
                    let c = &ma32 * &mb32;
                    std::hint::black_box(c[(0, 0)]);
                });
                let resident = gpu.prepare(n, &a, &b);
                resident.multiply();
                let gpu32 = time_ms(|| resident.multiply());
                let corner = gpu.read_corner(&resident.c_buf);
                std::hint::black_box(corner);
                eprintln!(
                    "n={n} cpu_f64_ms={cpu64} cpu_f32_ms={cpu32} gpu_f32_resident_ms={gpu32}"
                );
            }
        }
    }
}
