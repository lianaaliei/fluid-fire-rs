# Fluid Fire (Rust + wgpu)

Real-time 3D fire and smoke. A 64³ grid is simulated with Jos Stam's Stable Fluids,
entirely in WGSL compute shaders, and drawn with a volume ray marcher.

## Run

    cargo run --release

Needs a recent Rust toolchain (stable) and a GPU with Vulkan, Metal or DX12.

## Controls

| Input | Action |
| --- | --- |
| Left mouse drag | Orbit camera |
| Mouse wheel | Zoom |
| Right mouse drag | Paint fire and wind into the box |
| Q / A | Buoyancy up / down |
| W / S | Swirl up / down |
| E / D | Cooling up / down |
| R | Reset |
| Esc | Quit |

## Layout

- `src/main.rs` — window, wgpu setup, per-frame simulation + render loop, input
- `src/common.wgsl` — constants and the uniform struct shared by both shaders
- `src/sim.wgsl` — compute kernels: curl, forces, divergence, Jacobi, project, advect
- `src/render.wgsl` — volume ray marching with self-shadowing and fire emission

## Tweaking

`N` (grid size, multiple of 4), `ITER` (pressure iterations) and `SUBSTEPS` are constants at the
top of `main.rs`. Try `N = 96` or `128` on a strong GPU.

The project pins `wgpu 0.19` and `winit 0.29`; newer versions changed parts of the API.
