// 程序化生成的声音素材：房间脉冲响应、环境音、音效。
// 放同名文件到 web/sounds/ambience/<name>.(ogg|mp3|wav) 或 web/sounds/sfx/<name>.(...) 可替换成真实录音。

const EXTS = ["ogg", "mp3", "wav"];
let available = new Set(); // 后端列出的 web/sounds/ 下实际存在的文件

export function setAvailableSounds(paths) {
  available = new Set(paths);
}

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296 * 2 - 1;
  };
}

// 最简单的双二阶滤波器（RBJ cookbook），对 Float32Array 原地处理。
function biquad(data, type, freq, q, sr) {
  const w = 2 * Math.PI * freq / sr, cos = Math.cos(w), alpha = Math.sin(w) / (2 * q);
  let b0, b1, b2;
  if (type === "lowpass") { b0 = (1 - cos) / 2; b1 = 1 - cos; b2 = b0; }
  else if (type === "highpass") { b0 = (1 + cos) / 2; b1 = -(1 + cos); b2 = b0; }
  else { b0 = alpha; b1 = 0; b2 = -alpha; } // bandpass
  const a0 = 1 + alpha, a1 = -2 * cos, a2 = 1 - alpha;
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < data.length; i++) {
    const x = data[i];
    const y = (b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
    x2 = x1; x1 = x; y2 = y1; y1 = y;
    data[i] = y;
  }
  return data;
}

function pinkNoise(n, rand) {
  const out = new Float32Array(n);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < n; i++) {
    const w = rand();
    b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759;
    b2 = 0.969 * b2 + w * 0.153852; b3 = 0.8665 * b3 + w * 0.3104856;
    b4 = 0.55 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.016898;
    out[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11;
    b6 = w * 0.115926;
  }
  return out;
}

function normalize(data, peak = 0.8) {
  let max = 0;
  for (const v of data) max = Math.max(max, Math.abs(v));
  if (max > 0) for (let i = 0; i < data.length; i++) data[i] *= peak / max;
  return data;
}

function makeBuffer(ctx, channels) {
  const buf = ctx.createBuffer(channels.length, channels[0].length, ctx.sampleRate);
  channels.forEach((ch, i) => buf.copyToChannel(ch, i));
  return buf;
}

// 小房间 IR：早期反射 + 指数衰减的去相关噪声。
export function roomImpulse(ctx, seconds = 0.6, decay = 0.13) {
  const sr = ctx.sampleRate, n = Math.floor(seconds * sr);
  const channels = [0, 1].map((c) => {
    const rand = rng(7 + c * 101);
    const d = new Float32Array(n);
    for (let i = 0; i < n; i++) d[i] = rand() * Math.exp(-i / sr / decay);
    for (const [t, g] of [[0.007, 0.5], [0.013, 0.35], [0.021, 0.25]]) {
      const k = Math.floor((t + c * 0.0017) * sr);
      if (k < n) d[k] += g;
    }
    return normalize(biquad(d, "lowpass", 6000, 0.7, sr), 0.5);
  });
  return makeBuffer(ctx, channels);
}

const AMBIENCE = {
  rain(ctx) {
    const sr = ctx.sampleRate, n = sr * 12;
    return makeBuffer(ctx, [0, 1].map((c) => {
      const rand = rng(11 + c * 17);
      const bed = biquad(biquad(pinkNoise(n, rand), "lowpass", 5000, 0.6, sr), "highpass", 250, 0.7, sr);
      // 零星的雨滴：短促的带通噗声
      for (let k = 0; k < 900; k++) {
        const at = Math.floor(((rand() + 1) / 2) * (n - sr * 0.05));
        const len = Math.floor(sr * (0.004 + 0.01 * (rand() + 1) / 2));
        const amp = 0.15 + 0.35 * (rand() + 1) / 2;
        for (let i = 0; i < len; i++) bed[at + i] += rand() * amp * Math.exp(-i / (len / 4));
      }
      // 首尾交叉淡化，循环无缝
      const fade = sr;
      for (let i = 0; i < fade; i++) {
        const g = i / fade;
        bed[i] = bed[i] * g + bed[n - fade + i] * (1 - g);
      }
      return normalize(bed.subarray(0, n - fade), 0.5);
    }));
  },
  fire(ctx) {
    const sr = ctx.sampleRate, n = sr * 12;
    return makeBuffer(ctx, [0, 1].map((c) => {
      const rand = rng(23 + c * 5);
      const bed = biquad(pinkNoise(n, rand), "lowpass", 600, 0.7, sr);
      for (let i = 0; i < n; i++) bed[i] *= 0.6;
      for (let k = 0; k < 500; k++) {
        const at = Math.floor(((rand() + 1) / 2) * (n - sr * 0.02));
        const len = Math.floor(sr * 0.003);
        const amp = 0.3 + 0.7 * ((rand() + 1) / 2) ** 3;
        for (let i = 0; i < len; i++) bed[at + i] += rand() * amp * (1 - i / len);
      }
      const fade = sr;
      for (let i = 0; i < fade; i++) {
        const g = i / fade;
        bed[i] = bed[i] * g + bed[n - fade + i] * (1 - g);
      }
      return normalize(bed.subarray(0, n - fade), 0.5);
    }));
  },
};

function envelope(n, attack, release) {
  const env = new Float32Array(n);
  for (let i = 0; i < n; i++) env[i] = Math.min(1, i / attack, (n - i) / release);
  return env;
}

const SFX = {
  page_turn(ctx) {
    const sr = ctx.sampleRate, n = Math.floor(sr * 0.55), rand = rng(3);
    const d = new Float32Array(n);
    for (let i = 0; i < n; i++) d[i] = rand();
    // 纸张摩擦：带通中心频率从 2k 扫到 5k
    const out = new Float32Array(n);
    const seg = 256;
    for (let s = 0; s < n; s += seg) {
      const part = d.slice(s, s + seg);
      biquad(part, "bandpass", 2000 + 3000 * (s / n), 1.2, sr);
      out.set(part, s);
    }
    const env = envelope(n, sr * 0.03, sr * 0.25);
    for (let i = 0; i < n; i++) out[i] *= env[i] * (i < sr * 0.04 ? 2 : 1);
    return makeBuffer(ctx, [normalize(out, 0.5)]);
  },
  tea_pour(ctx) {
    const sr = ctx.sampleRate, n = Math.floor(sr * 2.8), rand = rng(9);
    const out = new Float32Array(n);
    const seg = 512;
    for (let s = 0; s < n; s += seg) {
      const part = new Float32Array(Math.min(seg, n - s)).map(() => rand());
      // 杯子越满音调越高
      biquad(part, "bandpass", 700 + 900 * (s / n), 3, sr);
      out.set(part, s);
    }
    const env = envelope(n, sr * 0.2, sr * 0.4);
    let gurgle = 0;
    for (let i = 0; i < n; i++) {
      gurgle = 0.999 * gurgle + 0.001 * rand();
      out[i] *= env[i] * (0.7 + 0.3 * Math.sin(2 * Math.PI * 9 * i / sr + gurgle * 40));
    }
    return makeBuffer(ctx, [normalize(out, 0.45)]);
  },
  fabric_rustle(ctx) {
    const sr = ctx.sampleRate, n = Math.floor(sr * 1.3), rand = rng(31);
    const out = biquad(new Float32Array(n).map(() => rand()), "bandpass", 1500, 0.5, sr);
    let amp = 0;
    for (let i = 0; i < n; i++) {
      if (i % 2000 === 0) amp = 0.3 + 0.7 * Math.abs(rand());
      out[i] *= amp * Math.sin(Math.PI * i / n);
    }
    return makeBuffer(ctx, [normalize(out, 0.4)]);
  },
  // 空间感测试用：轻刷麦克风似的短促摩擦声
  brush(ctx) {
    const sr = ctx.sampleRate, n = Math.floor(sr * 0.9), rand = rng(41);
    const out = biquad(new Float32Array(n).map(() => rand()), "bandpass", 3000, 0.8, sr);
    for (let i = 0; i < n; i++) out[i] *= Math.sin(Math.PI * i / n) * (0.6 + 0.4 * Math.sin(2 * Math.PI * 6 * i / sr));
    return makeBuffer(ctx, [normalize(out, 0.6)]);
  },
};

async function tryFetch(ctx, dir, name) {
  for (const ext of EXTS) {
    const path = `${dir}/${name}.${ext}`;
    if (!available.has(path)) continue;
    try {
      const res = await fetch(`sounds/${path}`);
      if (res.ok) return await ctx.decodeAudioData(await res.arrayBuffer());
    } catch (_) { /* 没有就用程序化素材 */ }
  }
  return null;
}

export async function loadAmbience(ctx, name) {
  if (!name || name === "none") return null;
  return (await tryFetch(ctx, "ambience", name)) || (AMBIENCE[name] ? AMBIENCE[name](ctx) : null);
}

export async function loadSfx(ctx, name) {
  return (await tryFetch(ctx, "sfx", name)) || (SFX[name] ? SFX[name](ctx) : null);
}

export const AMBIENCE_NAMES = Object.keys(AMBIENCE);
