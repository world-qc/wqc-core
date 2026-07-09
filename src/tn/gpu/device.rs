//! WebGPU compute acceleration for MPS tensor kernels (`wgpu`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use ndarray::{Array3, Array4};
use num_complex::Complex64;
use wgpu::util::DeviceExt;

use crate::engine::EngineError;
use crate::tn::gates::Mat2;

const SHADER_SRC: &str = include_str!("shaders.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vec2f {
    re: f32,
    im: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OneQubitParams {
    left: u32,
    right: u32,
    _pad0: u32,
    _pad1: u32,
    u00: Vec2f,
    u01: Vec2f,
    u10: Vec2f,
    u11: Vec2f,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MergeParams {
    dl: u32,
    dr: u32,
    bond: u32,
    _pad: u32,
}

/// Shared wgpu device for MPS kernels; tracks peak allocated buffer bytes (VRAM proxy).
pub struct GpuMpsDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    one_qubit_pipeline: wgpu::ComputePipeline,
    merge_pipeline: wgpu::ComputePipeline,
    peak_bytes: AtomicU64,
    one_qubit_layout: wgpu::BindGroupLayout,
    merge_layout: wgpu::BindGroupLayout,
}

impl GpuMpsDevice {
    /// Initialize WebGPU (Vulkan/Metal/DX12). Returns `None` if no adapter is available.
    pub fn try_new() -> Option<Arc<Self>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("wqc-mps"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wqc-mps-shaders"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let one_qubit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("one-qubit-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let merge_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("merge-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let one_qubit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apply-one-qubit"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("one-qubit-pipeline"),
                    bind_group_layouts: &[&one_qubit_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &module,
            entry_point: Some("apply_one_qubit"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let merge_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("merge-two-site"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("merge-pipeline"),
                    bind_group_layouts: &[&merge_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &module,
            entry_point: Some("merge_two_site"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Some(Arc::new(Self {
            device,
            queue,
            one_qubit_pipeline,
            merge_pipeline,
            peak_bytes: AtomicU64::new(0),
            one_qubit_layout,
            merge_layout,
        }))
    }

    pub fn peak_vram_bytes(&self) -> u64 {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    fn track_allocation(&self, bytes: u64) {
        let mut current = self.peak_bytes.load(Ordering::Relaxed);
        while bytes > current {
            match self.peak_bytes.compare_exchange_weak(
                current,
                bytes,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
    }

    /// Apply a 1-qubit unitary to one MPS site tensor on the GPU.
    pub fn apply_one_qubit(
        &self,
        site: &mut Array3<Complex64>,
        u: &Mat2,
    ) -> Result<(), EngineError> {
        let (left, _, right) = site.dim();
        let mut flat = site_to_vec2f(site);
        let params = OneQubitParams {
            left: left as u32,
            right: right as u32,
            _pad0: 0,
            _pad1: 0,
            u00: complex_to_vec2f(u[0][0]),
            u01: complex_to_vec2f(u[0][1]),
            u10: complex_to_vec2f(u[1][0]),
            u11: complex_to_vec2f(u[1][1]),
        };

        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("one-qubit-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let site_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("one-qubit-site"),
                contents: bytemuck::cast_slice(&flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        self.track_allocation((params_buf.size() + site_buf.size()) as u64);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("one-qubit-bind"),
            layout: &self.one_qubit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: site_buf.as_entire_binding(),
                },
            ],
        });

        let pairs = (left * right) as u32;
        let workgroups = (pairs + 255) / 256;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("one-qubit-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("one-qubit-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.one_qubit_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("one-qubit-readback"),
            size: site_buf.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&site_buf, 0, &readback, 0, site_buf.size());
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        flat = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();

        vec2f_to_site(&flat, site, left, right);
        Ok(())
    }

    /// Contract adjacent MPS sites into a two-qubit theta tensor on the GPU.
    pub fn merge_two_site(
        &self,
        left: &Array3<Complex64>,
        right: &Array3<Complex64>,
    ) -> Result<Array4<Complex64>, EngineError> {
        let dl = left.dim().0;
        let bond = left.dim().2.min(right.dim().0);
        let dr = right.dim().2;

        let left_flat = site_to_vec2f(left);
        let right_flat = site_to_vec2f(right);
        let theta_len = dl * 2 * 2 * dr;
        let mut theta_flat = vec![Vec2f { re: 0.0, im: 0.0 }; theta_len];

        let params = MergeParams {
            dl: dl as u32,
            dr: dr as u32,
            bond: bond as u32,
            _pad: 0,
        };

        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("merge-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let left_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("merge-left"),
                contents: bytemuck::cast_slice(&left_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let right_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("merge-right"),
                contents: bytemuck::cast_slice(&right_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let theta_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merge-theta"),
            size: (theta_len * std::mem::size_of::<Vec2f>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.track_allocation(
            (params_buf.size() + left_buf.size() + right_buf.size() + theta_buf.size()) as u64,
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("merge-bind"),
            layout: &self.merge_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: left_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: right_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: theta_buf.as_entire_binding(),
                },
            ],
        });

        let total = theta_len as u32;
        let workgroups = (total + 255) / 256;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("merge-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("merge-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.merge_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merge-readback"),
            size: theta_buf.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&theta_buf, 0, &readback, 0, theta_buf.size());
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        theta_flat = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();

        let mut theta = Array4::<Complex64>::zeros((dl, 2, 2, dr));
        for l in 0..dl {
            for s in 0..2 {
                for t in 0..2 {
                    for r in 0..dr {
                        let idx = l * 4 * dr + s * 2 * dr + t * dr + r;
                        let v = theta_flat[idx];
                        theta[[l, s, t, r]] = Complex64::new(v.re as f64, v.im as f64);
                    }
                }
            }
        }
        Ok(theta)
    }
}

fn complex_to_vec2f(c: Complex64) -> Vec2f {
    Vec2f {
        re: c.re as f32,
        im: c.im as f32,
    }
}

fn site_to_vec2f(site: &Array3<Complex64>) -> Vec<Vec2f> {
    let (left, phys, right) = site.dim();
    let mut out = Vec::with_capacity(left * phys * right);
    for a in 0..left {
        for s in 0..phys {
            for b in 0..right {
                out.push(complex_to_vec2f(site[[a, s, b]]));
            }
        }
    }
    out
}

fn vec2f_to_site(flat: &[Vec2f], site: &mut Array3<Complex64>, left: usize, right: usize) {
    for a in 0..left {
        for s in 0..2 {
            for b in 0..right {
                let idx = a * 2 * right + s * right + b;
                let v = flat[idx];
                site[[a, s, b]] = Complex64::new(v.re as f64, v.im as f64);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use num_complex::Complex64;

    #[test]
    fn gpu_one_qubit_matches_cpu_reference() {
        let Some(gpu) = GpuMpsDevice::try_new() else {
            eprintln!("skip gpu_one_qubit_matches_cpu_reference: no WebGPU adapter");
            return;
        };

        let u = [
            [
                Complex64::new(1.0 / 2.0f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0f64.sqrt(), 0.0),
            ],
            [
                Complex64::new(1.0 / 2.0f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0f64.sqrt(), 0.0),
            ],
        ];

        let mut cpu_site = Array3::<Complex64>::zeros((1, 2, 1));
        cpu_site[[0, 0, 0]] = Complex64::ONE;
        let mut gpu_site = cpu_site.clone();

        for a in 0..1 {
            for b in 0..1 {
                let v0 = cpu_site[[a, 0, b]];
                let v1 = cpu_site[[a, 1, b]];
                cpu_site[[a, 0, b]] = u[0][0] * v0 + u[0][1] * v1;
                cpu_site[[a, 1, b]] = u[1][0] * v0 + u[1][1] * v1;
            }
        }

        gpu.apply_one_qubit(&mut gpu_site, &u).expect("gpu apply");

        assert!((gpu_site[[0, 0, 0]].re - cpu_site[[0, 0, 0]].re).abs() < 1e-5);
        assert!((gpu_site[[0, 1, 0]].re - cpu_site[[0, 1, 0]].re).abs() < 1e-5);
    }
}
