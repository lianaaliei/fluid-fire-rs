@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read> scS: array<vec2<f32>>;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4<f32>(corners[vi], 0.0, 1.0);
  o.uv = corners[vi];
  return o;
}

fn scAt(p: vec3<i32>) -> vec2<f32> {
  if (p.x < 0 || p.y < 0 || p.z < 0 || p.x >= N || p.y >= N || p.z >= N) { return vec2<f32>(0.0); }
  return scS[u32(p.x + p.y * N + p.z * N * N)];
}

fn sampleSc(x: vec3<f32>) -> vec2<f32> {
  let q = x - vec3<f32>(0.5);
  let fl = floor(q);
  let b = vec3<i32>(fl);
  let f = q - fl;
  let c00 = mix(scAt(b), scAt(b + vec3<i32>(1, 0, 0)), f.x);
  let c10 = mix(scAt(b + vec3<i32>(0, 1, 0)), scAt(b + vec3<i32>(1, 1, 0)), f.x);
  let c01 = mix(scAt(b + vec3<i32>(0, 0, 1)), scAt(b + vec3<i32>(1, 0, 1)), f.x);
  let c11 = mix(scAt(b + vec3<i32>(0, 1, 1)), scAt(b + vec3<i32>(1, 1, 1)), f.x);
  return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

fn hash12(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs(inp: VOut) -> @location(0) vec4<f32> {
  let ro = U.eye.xyz;
  let rd = normalize(U.fwd.xyz + inp.uv.x * U.eye.w * U.fwd.w * U.right.xyz + inp.uv.y * U.fwd.w * U.up.xyz);

  var bg = mix(vec3<f32>(0.012, 0.014, 0.024), vec3<f32>(0.05, 0.06, 0.09), inp.uv.y * 0.5 + 0.5);
  bg = bg * (1.0 - 0.25 * dot(inp.uv, inp.uv) * 0.5);

  // ray vs. unit box centred at the origin
  let inv = 1.0 / rd;
  let t0 = (vec3<f32>(-0.5) - ro) * inv;
  let t1 = (vec3<f32>(0.5) - ro) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let tn = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
  let tf = min(min(tmax.x, tmax.y), tmax.z);
  if (tf <= tn) { return vec4<f32>(pow(bg, vec3<f32>(1.0 / 2.2)), 1.0); }

  let nsteps = 90;
  let dt = (tf - tn) / f32(nsteps);
  let jit = hash12(inp.pos.xy + vec2<f32>(U.a.y * 37.0, U.a.y * 11.0));
  let L = normalize(vec3<f32>(0.5, 0.8, 0.35));

  var T = 1.0;
  var col = vec3<f32>(0.0);
  for (var i = 0; i < nsteps; i += 1) {
    let pos = ro + rd * (tn + (f32(i) + jit) * dt);
    let s = sampleSc((pos + vec3<f32>(0.5)) * NF);
    let sigma = s.x * U.c.x;
    if (sigma > 0.02) {
      // short march toward the light for self-shadowing
      var tau = 0.0;
      for (var k = 1; k <= 4; k += 1) {
        let lp = pos + L * (0.05 * f32(k));
        tau += sampleSc((lp + vec3<f32>(0.5)) * NF).x;
      }
      let lit = exp(-tau * U.c.x * 0.05);
      let smoke = vec3<f32>(0.55, 0.57, 0.62) * (vec3<f32>(0.10, 0.12, 0.17) + vec3<f32>(1.5, 1.35, 1.15) * lit);
      let heat = s.y;
      let fire = vec3<f32>(1.0, 0.33, 0.07) * pow(heat, 1.5) * 5.0
               + vec3<f32>(1.0, 0.78, 0.35) * pow(heat, 4.0) * 7.0;
      let a = 1.0 - exp(-sigma * dt);
      col += T * a * (smoke * 0.5 + fire * U.c.y);
      T *= 1.0 - a;
      if (T < 0.02) { break; }
    }
  }

  var outc = col + T * bg;
  outc = vec3<f32>(1.0) - exp(-outc * 1.35);
  outc = pow(outc, vec3<f32>(1.0 / 2.2));
  return vec4<f32>(outc, 1.0);
}
