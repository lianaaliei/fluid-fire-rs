@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read> velS: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velD: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> scS: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> scD: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> pS: array<f32>;
@group(0) @binding(6) var<storage, read_write> pD: array<f32>;
@group(0) @binding(7) var<storage, read_write> divB: array<f32>;
@group(0) @binding(8) var<storage, read_write> curlB: array<vec4<f32>>;

fn idx(p: vec3<i32>) -> u32 {
  let c = clamp(p, vec3<i32>(0), vec3<i32>(N - 1));
  return u32(c.x + c.y * N + c.z * N * N);
}

// Solid walls on five sides, open ceiling.
fn velAt(p: vec3<i32>) -> vec3<f32> {
  if (p.x < 0 || p.x >= N || p.z < 0 || p.z >= N || p.y < 0) { return vec3<f32>(0.0); }
  return velS[idx(vec3<i32>(p.x, min(p.y, N - 1), p.z))].xyz;
}

fn pAt(p: vec3<i32>) -> f32 {
  if (p.y >= N) { return 0.0; }
  return pS[idx(p)];
}

fn scAt(p: vec3<i32>) -> vec2<f32> { return scS[idx(p)]; }

fn sampleVel(x: vec3<f32>) -> vec3<f32> {
  let q = x - vec3<f32>(0.5);
  let fl = floor(q);
  let b = vec3<i32>(fl);
  let f = q - fl;
  let c00 = mix(velAt(b), velAt(b + vec3<i32>(1, 0, 0)), f.x);
  let c10 = mix(velAt(b + vec3<i32>(0, 1, 0)), velAt(b + vec3<i32>(1, 1, 0)), f.x);
  let c01 = mix(velAt(b + vec3<i32>(0, 0, 1)), velAt(b + vec3<i32>(1, 0, 1)), f.x);
  let c11 = mix(velAt(b + vec3<i32>(0, 1, 1)), velAt(b + vec3<i32>(1, 1, 1)), f.x);
  return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
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

@compute @workgroup_size(4, 4, 4)
fn curlK(@builtin(global_invocation_id) g: vec3<u32>) {
  let p = vec3<i32>(g);
  let xp = velAt(p + vec3<i32>(1, 0, 0));
  let xm = velAt(p - vec3<i32>(1, 0, 0));
  let yp = velAt(p + vec3<i32>(0, 1, 0));
  let ym = velAt(p - vec3<i32>(0, 1, 0));
  let zp = velAt(p + vec3<i32>(0, 0, 1));
  let zm = velAt(p - vec3<i32>(0, 0, 1));
  let w = 0.5 * vec3<f32>(
    (yp.z - ym.z) - (zp.y - zm.y),
    (zp.x - zm.x) - (xp.z - xm.z),
    (xp.y - xm.y) - (yp.x - ym.x));
  curlB[idx(p)] = vec4<f32>(w, length(w));
}

@compute @workgroup_size(4, 4, 4)
fn forcesK(@builtin(global_invocation_id) g: vec3<u32>) {
  let p = vec3<i32>(g);
  let i = idx(p);
  var v = velS[i].xyz;
  var s = scS[i];
  let cc = vec3<f32>(p) + vec3<f32>(0.5);
  let dt = U.a.x;

  // vorticity confinement: put back the small swirls that numerical diffusion eats
  let eta = 0.5 * vec3<f32>(
    curlB[idx(p + vec3<i32>(1, 0, 0))].w - curlB[idx(p - vec3<i32>(1, 0, 0))].w,
    curlB[idx(p + vec3<i32>(0, 1, 0))].w - curlB[idx(p - vec3<i32>(0, 1, 0))].w,
    curlB[idx(p + vec3<i32>(0, 0, 1))].w - curlB[idx(p - vec3<i32>(0, 0, 1))].w);
  let el = length(eta);
  if (el > 0.00001) {
    v += U.a.w * dt * cross(eta / el, curlB[i].xyz);
  }

  // hot gas rises
  v.y += dt * U.a.z * s.y;

  // emitter
  let de = distance(cc, U.emit.xyz);
  if (de < U.emit.w) {
    let k = 1.0 - de / U.emit.w;
    let kk = k * k * (3.0 - 2.0 * k);
    s.x = min(s.x + kk * U.b.x, 2.5);
    s.y = max(s.y, kk * 1.3);
    v.y += kk * 0.05;
    v.x += kk * 0.03 * sin(U.a.y * 2.3 + cc.y * 0.4 + cc.z * 0.3);
    v.z += kk * 0.03 * cos(U.a.y * 1.9 + cc.y * 0.35 + cc.x * 0.3);
  }

  // pointer impulse
  if (U.impv.w > 0.5) {
    let di = distance(cc, U.imp.xyz);
    if (di < U.imp.w) {
      let k = 1.0 - di / U.imp.w;
      let kk = k * k;
      v += U.impv.xyz * kk;
      s.x = min(s.x + kk * 0.12, 2.5);
      s.y = max(s.y, kk * 0.9);
    }
  }

  velD[i] = vec4<f32>(v, 0.0);
  scD[i] = clamp(s, vec2<f32>(0.0), vec2<f32>(3.0, 2.0));
}

@compute @workgroup_size(4, 4, 4)
fn divK(@builtin(global_invocation_id) g: vec3<u32>) {
  let p = vec3<i32>(g);
  let d = 0.5 * ((velAt(p + vec3<i32>(1, 0, 0)).x - velAt(p - vec3<i32>(1, 0, 0)).x)
               + (velAt(p + vec3<i32>(0, 1, 0)).y - velAt(p - vec3<i32>(0, 1, 0)).y)
               + (velAt(p + vec3<i32>(0, 0, 1)).z - velAt(p - vec3<i32>(0, 0, 1)).z));
  divB[idx(p)] = d;
}

@compute @workgroup_size(4, 4, 4)
fn jacobiK(@builtin(global_invocation_id) g: vec3<u32>) {
  let p = vec3<i32>(g);
  let sum = pAt(p + vec3<i32>(1, 0, 0)) + pAt(p - vec3<i32>(1, 0, 0))
          + pAt(p + vec3<i32>(0, 1, 0)) + pAt(p - vec3<i32>(0, 1, 0))
          + pAt(p + vec3<i32>(0, 0, 1)) + pAt(p - vec3<i32>(0, 0, 1));
  pD[idx(p)] = (sum - divB[idx(p)]) / 6.0;
}

@compute @workgroup_size(4, 4, 4)
fn projectK(@builtin(global_invocation_id) g: vec3<u32>) {
  let p = vec3<i32>(g);
  let i = idx(p);
  var v = velS[i].xyz;
  v -= 0.5 * vec3<f32>(
    pAt(p + vec3<i32>(1, 0, 0)) - pAt(p - vec3<i32>(1, 0, 0)),
    pAt(p + vec3<i32>(0, 1, 0)) - pAt(p - vec3<i32>(0, 1, 0)),
    pAt(p + vec3<i32>(0, 0, 1)) - pAt(p - vec3<i32>(0, 0, 1)));
  if (p.x == 0 || p.x == N - 1) { v.x = 0.0; }
  if (p.z == 0 || p.z == N - 1) { v.z = 0.0; }
  if (p.y == 0) { v.y = 0.0; }
  velD[i] = vec4<f32>(v, 0.0);
}

@compute @workgroup_size(4, 4, 4)
fn advectK(@builtin(global_invocation_id) g: vec3<u32>) {
  let p = vec3<i32>(g);
  let i = idx(p);
  let cc = vec3<f32>(p) + vec3<f32>(0.5);
  let v = velS[i].xyz;
  let back = clamp(cc - U.a.x * v, vec3<f32>(0.5), vec3<f32>(NF - 0.5));
  velD[i] = vec4<f32>(sampleVel(back) * U.b.w, 0.0);
  let r = sampleSc(back);
  scD[i] = vec2<f32>(r.x * (1.0 - U.b.y), r.y * (1.0 - U.b.z));
}
