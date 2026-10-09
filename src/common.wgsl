const N: i32 = {{N}};
const NF: f32 = {{N}}.0;

struct Params {
  a: vec4<f32>,      // dt, time, buoyancy, swirl
  emit: vec4<f32>,   // emitter xyz (cells), radius
  b: vec4<f32>,      // emit strength, density loss, cooling, velocity drag
  imp: vec4<f32>,    // impulse position xyz, radius
  impv: vec4<f32>,   // impulse velocity xyz, on/off
  eye: vec4<f32>,    // camera position, aspect
  fwd: vec4<f32>,    // camera forward, tan(fov/2)
  right: vec4<f32>,
  up: vec4<f32>,
  c: vec4<f32>,      // extinction, fire gain
}
