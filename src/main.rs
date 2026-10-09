//! Fluid Fire — real-time 3D "Stable Fluids" (Jos Stam) smoke and fire on the GPU.
//!
//! Every simulation step runs as WGSL compute shaders through wgpu:
//!   curl -> forces (vorticity + buoyancy + emitter) -> divergence
//!   -> Jacobi pressure solve -> projection -> semi-Lagrangian advection.
//! The result is drawn with a volume ray marcher in a fragment shader.
//!
//! Controls
//!   left mouse drag   orbit camera          mouse wheel   zoom
//!   right mouse drag  paint fire and wind   R             reset
//!   Q / A             buoyancy up / down    W / S         swirl up / down
//!   E / D             cooling up / down     Esc           quit

use std::sync::Arc;
use std::time::Instant;

use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowBuilder};

/// Grid resolution: the simulation box is N x N x N cells. Must be a multiple of 4.
const N: u32 = 64;
/// Jacobi iterations for the pressure solve, per simulation step.
const ITER: usize = 28;
/// Simulation steps per rendered frame.
const SUBSTEPS: usize = 2;
/// tan(fov / 2) of the camera.
const TAN_HALF: f32 = 0.714;

// ---------------------------------------------------------------------------
// tiny vector helpers
// ---------------------------------------------------------------------------
type V3 = [f32; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn length(a: V3) -> f32 {
    dot(a, a).sqrt()
}
fn normalize(a: V3) -> V3 {
    scale(a, 1.0 / length(a))
}

// ---------------------------------------------------------------------------
// camera
// ---------------------------------------------------------------------------
struct Camera {
    yaw: f32,
    pitch: f32,
    dist: f32,
}

impl Camera {
    /// Returns (eye, forward, right, up). The fluid box is centred on the origin, side length 1.
    fn vectors(&self) -> (V3, V3, V3, V3) {
        let cp = self.pitch.cos();
        let eye = [
            self.dist * cp * self.yaw.sin(),
            self.dist * self.pitch.sin(),
            self.dist * cp * self.yaw.cos(),
        ];
        let fwd = normalize(scale(eye, -1.0));
        let right = normalize(cross(fwd, [0.0, 1.0, 0.0]));
        let up = cross(right, fwd);
        (eye, fwd, right, up)
    }
}

// ---------------------------------------------------------------------------
// GPU side
// ---------------------------------------------------------------------------
struct Pipelines {
    curl: wgpu::ComputePipeline,
    forces: wgpu::ComputePipeline,
    div: wgpu::ComputePipeline,
    jacobi: wgpu::ComputePipeline,
    project: wgpu::ComputePipeline,
    advect: wgpu::ComputePipeline,
}

struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    pipes: Pipelines,
    render_pipe: wgpu::RenderPipeline,

    ubuf: wgpu::Buffer,
    vel: [wgpu::Buffer; 2],
    sc: [wgpu::Buffer; 2], // (smoke density, temperature)
    pr: [wgpu::Buffer; 2],
    div: wgpu::Buffer,
    curl: wgpu::Buffer,

    sim_bgs: Vec<wgpu::BindGroup>, // indexed by vi*4 + si*2 + pi
    render_bgs: Vec<wgpu::BindGroup>, // indexed by si

    // which half of each ping-pong pair currently holds the "source" data
    vi: usize,
    si: usize,
    pi: usize,
}

fn shader_source(body: &str) -> String {
    format!("{}\n{}", include_str!("common.wgsl"), body).replace("{{N}}", &N.to_string())
}

fn storage_buffer(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn layout_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl Gpu {
    async fn new(window: Arc<Window>) -> Gpu {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("could not create a window surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");
        println!("GPU: {}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("fluid-fire device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("could not open the GPU device");

        // The fragment shader already applies gamma, so draw to a non-sRGB surface format.
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // ---- buffers -------------------------------------------------------------------
        let cells = (N as u64).pow(3);
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 160,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vel = [
            storage_buffer(&device, "vel0", cells * 16),
            storage_buffer(&device, "vel1", cells * 16),
        ];
        let sc = [
            storage_buffer(&device, "sc0", cells * 8),
            storage_buffer(&device, "sc1", cells * 8),
        ];
        let pr = [
            storage_buffer(&device, "p0", cells * 4),
            storage_buffer(&device, "p1", cells * 4),
        ];
        let div = storage_buffer(&device, "div", cells * 4);
        let curl = storage_buffer(&device, "curl", cells * 16);

        // ---- simulation pipelines --------------------------------------------------------
        let sim_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sim"),
            source: wgpu::ShaderSource::Wgsl(shader_source(include_str!("sim.wgsl")).into()),
        });

        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let c = wgpu::ShaderStages::COMPUTE;
        let sim_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim bgl"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform, c),
                layout_entry(1, ro, c),
                layout_entry(2, rw, c),
                layout_entry(3, ro, c),
                layout_entry(4, rw, c),
                layout_entry(5, ro, c),
                layout_entry(6, rw, c),
                layout_entry(7, rw, c),
                layout_entry(8, rw, c),
            ],
        });
        let sim_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim layout"),
            bind_group_layouts: &[&sim_bgl],
            push_constant_ranges: &[],
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&sim_layout),
                module: &sim_module,
                entry_point: entry,
            })
        };
        let pipes = Pipelines {
            curl: make("curlK"),
            forces: make("forcesK"),
            div: make("divK"),
            jacobi: make("jacobiK"),
            project: make("projectK"),
            advect: make("advectK"),
        };

        // One bind group for every combination of ping-pong parities.
        let mut sim_bgs = Vec::new();
        for vi in 0..2 {
            for si in 0..2 {
                for pi in 0..2 {
                    sim_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("sim bg"),
                        layout: &sim_bgl,
                        entries: &[
                            bind_entry(0, &ubuf),
                            bind_entry(1, &vel[vi]),
                            bind_entry(2, &vel[1 - vi]),
                            bind_entry(3, &sc[si]),
                            bind_entry(4, &sc[1 - si]),
                            bind_entry(5, &pr[pi]),
                            bind_entry(6, &pr[1 - pi]),
                            bind_entry(7, &div),
                            bind_entry(8, &curl),
                        ],
                    }));
                }
            }
        }

        // ---- render pipeline -----------------------------------------------------------
        let render_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render"),
            source: wgpu::ShaderSource::Wgsl(shader_source(include_str!("render.wgsl")).into()),
        });
        let f = wgpu::ShaderStages::FRAGMENT;
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render bgl"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform, f),
                layout_entry(1, ro, f),
            ],
        });
        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render layout"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });
        let render_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume render"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &render_module,
                entry_point: "vs",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_module,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        let render_bgs = (0..2)
            .map(|si| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("render bg"),
                    layout: &render_bgl,
                    entries: &[bind_entry(0, &ubuf), bind_entry(1, &sc[si])],
                })
            })
            .collect();

        Gpu {
            surface,
            device,
            queue,
            config,
            pipes,
            render_pipe,
            ubuf,
            vel,
            sc,
            pr,
            div,
            curl,
            sim_bgs,
            render_bgs,
            vi: 0,
            si: 0,
            pi: 0,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Runs the simulation for one frame and draws it.
    fn frame(&mut self, uniforms: &[f32; 40], reset: bool) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.queue
            .write_buffer(&self.ubuf, 0, bytemuck::cast_slice(uniforms));

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        if reset {
            for b in self
                .vel
                .iter()
                .chain(self.sc.iter())
                .chain(self.pr.iter())
                .chain([&self.div, &self.curl])
            {
                enc.clear_buffer(b, 0, None);
            }
            self.vi = 0;
            self.si = 0;
            self.pi = 0;
        }

        // ---- simulation ------------------------------------------------------------------
        let (mut vi, mut si, mut pi) = (self.vi, self.si, self.pi);
        {
            let wg = N / 4;
            let bgs = &self.sim_bgs;
            let pp = &self.pipes;
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stable fluids"),
                timestamp_writes: None,
            });
            macro_rules! run {
                ($pipeline:expr) => {{
                    cp.set_pipeline($pipeline);
                    cp.set_bind_group(0, &bgs[vi * 4 + si * 2 + pi], &[]);
                    cp.dispatch_workgroups(wg, wg, wg);
                }};
            }
            for _ in 0..SUBSTEPS {
                run!(&pp.curl);
                run!(&pp.forces);
                vi ^= 1;
                si ^= 1;
                run!(&pp.div);
                for _ in 0..ITER {
                    run!(&pp.jacobi);
                    pi ^= 1;
                }
                run!(&pp.project);
                vi ^= 1;
                run!(&pp.advect);
                vi ^= 1;
                si ^= 1;
            }
        }
        self.vi = vi;
        self.si = si;
        self.pi = pi;

        // ---- draw ------------------------------------------------------------------------
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volume"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(&self.render_pipe);
            rp.set_bind_group(0, &self.render_bgs[self.si], &[]);
            rp.draw(0..3, 0..1);
        }

        self.queue.submit(Some(enc.finish()));
        output.present();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// application state + input
// ---------------------------------------------------------------------------
struct App {
    window: Arc<Window>,
    gpu: Gpu,

    cam: Camera,
    buoy: f32,
    swirl: f32,
    cool: f32,
    time: f32,

    left: bool,
    right: bool,
    cursor: (f64, f64),
    last_grid: Option<V3>,
    impulse: Option<(V3, V3)>, // (grid position, velocity) applied next frame
    reset: bool,

    fps_clock: Instant,
    fps_frames: u32,
}

fn put(u: &mut [f32; 40], offset: usize, v: [f32; 4]) {
    u[offset..offset + 4].copy_from_slice(&v);
}

impl App {
    async fn new(window: Arc<Window>) -> App {
        let gpu = Gpu::new(window.clone()).await;
        App {
            window,
            gpu,
            cam: Camera { yaw: 0.6, pitch: 0.28, dist: 1.85 },
            buoy: 0.011,
            swirl: 0.35,
            cool: 0.010,
            time: 0.0,
            left: false,
            right: false,
            cursor: (0.0, 0.0),
            last_grid: None,
            impulse: None,
            reset: false,
            fps_clock: Instant::now(),
            fps_frames: 0,
        }
    }

    /// Where the mouse pointer lands on a plane through the box centre that faces the camera,
    /// in grid-cell coordinates (kept a few cells away from the walls).
    fn pointer_to_grid(&self, px: f64, py: f64) -> V3 {
        let w = self.gpu.config.width as f32;
        let h = self.gpu.config.height as f32;
        let nx = px as f32 / w * 2.0 - 1.0;
        let ny = 1.0 - py as f32 / h * 2.0;
        let aspect = w / h;
        let (eye, fwd, right, up) = self.cam.vectors();
        let dir = normalize(add(
            fwd,
            add(scale(right, nx * aspect * TAN_HALF), scale(up, ny * TAN_HALF)),
        ));
        let t = -dot(eye, fwd) / dot(dir, fwd);
        let p = add(eye, scale(dir, t));
        let n = N as f32;
        [
            ((p[0] + 0.5) * n).clamp(3.0, n - 3.0),
            ((p[1] + 0.5) * n).clamp(3.0, n - 3.0),
            ((p[2] + 0.5) * n).clamp(3.0, n - 3.0),
        ]
    }

    fn build_uniforms(&mut self) -> [f32; 40] {
        let t = self.time;
        let n = N as f32;
        let (eye, fwd, right, up) = self.cam.vectors();
        let aspect = self.gpu.config.width as f32 / self.gpu.config.height as f32;

        let mut u = [0.0f32; 40];
        put(&mut u, 0, [1.0, t, self.buoy, self.swirl]);
        // emitter wanders a little so the plume never looks mechanical
        put(
            &mut u,
            4,
            [n / 2.0 + (t * 0.37).sin() * 5.0, 6.0, n / 2.0 + (t * 0.29).cos() * 5.0, 6.5],
        );
        put(&mut u, 8, [0.30, 0.0015, self.cool, 0.97]);
        match self.impulse.take() {
            Some((p, v)) => {
                put(&mut u, 12, [p[0], p[1], p[2], 5.5]);
                put(&mut u, 16, [v[0], v[1], v[2], 1.0]);
            }
            None => {
                put(&mut u, 12, [0.0, 0.0, 0.0, 1.0]);
                put(&mut u, 16, [0.0, 0.0, 0.0, 0.0]);
            }
        }
        put(&mut u, 20, [eye[0], eye[1], eye[2], aspect]);
        put(&mut u, 24, [fwd[0], fwd[1], fwd[2], TAN_HALF]);
        put(&mut u, 28, [right[0], right[1], right[2], 0.0]);
        put(&mut u, 32, [up[0], up[1], up[2], 0.0]);
        put(&mut u, 36, [9.0, 1.0, 0.0, 0.0]);
        u
    }

    fn update_title(&mut self) {
        self.fps_frames += 1;
        let elapsed = self.fps_clock.elapsed().as_secs_f32();
        if elapsed >= 0.5 {
            let fps = self.fps_frames as f32 / elapsed;
            self.window.set_title(&format!(
                "Fluid Fire  |  {:.0} fps  |  {}^3 grid  |  buoyancy {:.3}  swirl {:.2}  cooling {:.3}",
                fps, N, self.buoy, self.swirl, self.cool
            ));
            self.fps_clock = Instant::now();
            self.fps_frames = 0;
        }
    }

    fn handle_key(&mut self, key: &Key) -> bool {
        match key {
            Key::Named(NamedKey::Escape) => return true,
            Key::Character(c) => match c.to_lowercase().as_str() {
                "r" => self.reset = true,
                "q" => self.buoy = (self.buoy + 0.0005).min(0.03),
                "a" => self.buoy = (self.buoy - 0.0005).max(0.0),
                "w" => self.swirl = (self.swirl + 0.05).min(1.0),
                "s" => self.swirl = (self.swirl - 0.05).max(0.0),
                "e" => self.cool = (self.cool + 0.001).min(0.03),
                "d" => self.cool = (self.cool - 0.001).max(0.002),
                _ => {}
            },
            _ => {}
        }
        false
    }

    fn cursor_moved(&mut self, x: f64, y: f64) {
        if self.left {
            let dx = (x - self.cursor.0) as f32;
            let dy = (y - self.cursor.1) as f32;
            self.cam.yaw -= dx * 0.006;
            self.cam.pitch = (self.cam.pitch + dy * 0.006).clamp(-1.1, 1.3);
        }
        if self.right {
            let g = self.pointer_to_grid(x, y);
            if let Some(prev) = self.last_grid {
                let mut v = [
                    (g[0] - prev[0]) * 0.1,
                    (g[1] - prev[1]) * 0.1,
                    (g[2] - prev[2]) * 0.1,
                ];
                let m = length(v);
                if m > 1.2 {
                    v = scale(v, 1.2 / m);
                }
                self.impulse = Some((g, v));
            }
            self.last_grid = Some(g);
        }
        self.cursor = (x, y);
    }

    fn render(&mut self, elwt: &winit::event_loop::EventLoopWindowTarget<()>) {
        self.time += 0.016;
        let uniforms = self.build_uniforms();
        let reset = std::mem::take(&mut self.reset);
        match self.gpu.frame(&uniforms, reset) {
            Ok(()) => self.update_title(),
            Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                let (w, h) = (self.gpu.config.width, self.gpu.config.height);
                self.gpu.resize(w, h);
            }
            Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
            Err(e) => eprintln!("surface error: {e:?}"),
        }
    }
}

fn main() {
    println!("Fluid Fire — left drag: orbit | right drag: paint fire | wheel: zoom");
    println!("Q/A buoyancy  W/S swirl  E/D cooling  R reset  Esc quit");

    let event_loop = EventLoop::new().expect("could not create the event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Fluid Fire")
            .with_inner_size(LogicalSize::new(1100.0, 760.0))
            .build(&event_loop)
            .expect("could not create the window"),
    );
    let mut app = pollster::block_on(App::new(window));

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);
            match event {
                Event::WindowEvent { event, window_id } if window_id == app.window.id() => {
                    match event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::Resized(size) => app.gpu.resize(size.width, size.height),
                        WindowEvent::RedrawRequested => app.render(elwt),
                        WindowEvent::CursorMoved { position, .. } => {
                            app.cursor_moved(position.x, position.y)
                        }
                        WindowEvent::MouseInput { state, button, .. } => {
                            let down = state == ElementState::Pressed;
                            match button {
                                MouseButton::Left => app.left = down,
                                MouseButton::Right => {
                                    app.right = down;
                                    app.last_grid = if down {
                                        Some(app.pointer_to_grid(app.cursor.0, app.cursor.1))
                                    } else {
                                        None
                                    };
                                }
                                _ => {}
                            }
                        }
                        WindowEvent::MouseWheel { delta, .. } => {
                            let dy = match delta {
                                MouseScrollDelta::LineDelta(_, y) => y,
                                MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                            };
                            app.cam.dist = (app.cam.dist * (1.0 - dy * 0.08)).clamp(1.1, 3.2);
                        }
                        WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    logical_key,
                                    state: ElementState::Pressed,
                                    ..
                                },
                            ..
                        } => {
                            if app.handle_key(&logical_key) {
                                elwt.exit();
                            }
                        }
                        _ => {}
                    }
                }
                Event::AboutToWait => app.window.request_redraw(),
                _ => {}
            }
        })
        .expect("event loop error");
}
