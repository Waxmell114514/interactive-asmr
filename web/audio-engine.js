// 音频前端：播放队列、HRTF 定位、近场效果、环境音、响度限制。
//
// 人声链路（Spec「音频与空间化」）：
//   PCM(24kHz mono) → level(独白衰减) → fade(打断淡出) → lowshelf 200Hz(近讲效应)
//     → distance gain → PannerNode(HRTF) → dry + Convolver(小房间, wet) → master → 限幅 -3dBFS
// 环境音：立体声循环 → 独立 gain(默认 −18dB) → master，不过 HRTF。
// 坐标：听者在原点、面向 −z；left = −x，right = +x，behind = +z。

import { loadAmbience, loadSfx, roomImpulse } from "./sounds.js";

export const TTS_RATE = 24000;

export const DISTANCES = {
  near: { r: 0.1, shelfDb: 4, wet: 0.03, gainDb: 0 },
  mid: { r: 0.5, shelfDb: 1, wet: 0.08, gainDb: -4 },
  far: { r: 1.5, shelfDb: 0, wet: 0.15, gainDb: -9 },
};
export const AZIMUTHS = { front: 0, right: 90, behind: 180, left: -90 };

const AMBIENCE_DB = -18;
const TC = 0.08; // 位置跳变的平滑时间常数
const FADE_SECS = 0.12; // 打断淡出
const LEAD = 0.12; // 空闲后首段的预缓冲
const GAP = 0.12; // 段落之间留一点呼吸

const dbToGain = (db) => Math.pow(10, db / 20);

export function coords(azimuthDeg, r) {
  const a = (azimuthDeg * Math.PI) / 180;
  return { x: r * Math.sin(a), z: -r * Math.cos(a) };
}

// 从 a 到 b 的最短角度路径（正好 180° 时经过正前方）。
export function arcPath(fromDeg, toDeg) {
  let diff = toDeg - fromDeg;
  if (diff > 180) diff -= 360;
  if (diff < -180) diff += 360;
  if (Math.abs(diff) === 180) diff = fromDeg > 0 ? -Math.abs(diff) : Math.abs(diff);
  return diff;
}

export class AudioEngine {
  constructor({ reportStart, reportEnd } = {}) {
    this.reportStart = reportStart || (() => {}); // 段落开始播放（回报 seg_start）
    this.reportEnd = reportEnd || (() => {}); // 段落播完（回报 seg_end）
    this.ctx = null;
    this.segments = new Map();
    this.cursor = 0;
    this.sources = new Set();
    this.timers = new Set();
    this.pos = { azimuth: "front", distance: "mid" };
    this.sfxCache = new Map();
    this.ambienceSource = null;
    this.ambienceLevel = 1;
  }

  async init() {
    if (this.ctx) return;
    const ctx = (this.ctx = new AudioContext({ latencyHint: "playback" }));

    this.master = new GainNode(ctx, { gain: 1 });
    this.limiter = new DynamicsCompressorNode(ctx, { threshold: -3, knee: 0, ratio: 20, attack: 0.003, release: 0.25 });
    this.master.connect(this.limiter).connect(ctx.destination);

    this.voiceVolume = new GainNode(ctx, { gain: 1 });
    this.level = new GainNode(ctx, { gain: 1 });
    this.fade = new GainNode(ctx, { gain: 1 });
    this.shelf = new BiquadFilterNode(ctx, { type: "lowshelf", frequency: 200, gain: 0 });
    this.distGain = new GainNode(ctx, { gain: 1 });
    this.panner = new PannerNode(ctx, {
      panningModel: "HRTF",
      distanceModel: "inverse",
      rolloffFactor: 0, // 距离衰减由 distGain 控制
      positionX: 0, positionY: 0, positionZ: -0.5,
    });
    this.dry = new GainNode(ctx, { gain: 1 });
    this.wet = new GainNode(ctx, { gain: 0.08 });
    this.reverb = new ConvolverNode(ctx, { buffer: roomImpulse(ctx) });
    this.voiceIn = this.level;
    this.level.connect(this.fade).connect(this.shelf).connect(this.distGain).connect(this.panner);
    this.panner.connect(this.dry).connect(this.voiceVolume);
    this.panner.connect(this.wet).connect(this.reverb).connect(this.voiceVolume);
    this.voiceVolume.connect(this.master);

    this.ambienceVolume = new GainNode(ctx, { gain: 1 });
    this.ambienceGain = new GainNode(ctx, { gain: 0 });
    this.ambienceGain.connect(this.ambienceVolume).connect(this.master);

    this.applyPosition(this.pos.azimuth, this.pos.distance, ctx.currentTime, 0.001);
  }

  // ---- 环境音 -------------------------------------------------------------

  async startAmbience(name, fadeSecs = 4) {
    const ctx = this.ctx;
    this.stopAmbience(0.5);
    const buffer = await loadAmbience(ctx, name);
    if (!buffer) return;
    const src = new AudioBufferSourceNode(ctx, { buffer, loop: true });
    src.connect(this.ambienceGain);
    src.start();
    this.ambienceSource = src;
    this.ambienceLevel = 1;
    const now = ctx.currentTime;
    this.ambienceGain.gain.cancelScheduledValues(now);
    this.ambienceGain.gain.setValueAtTime(0.0001, now);
    this.ambienceGain.gain.exponentialRampToValueAtTime(dbToGain(AMBIENCE_DB), now + fadeSecs);
  }

  stopAmbience(fadeSecs = 1) {
    const src = this.ambienceSource;
    if (!src) return;
    this.ambienceSource = null;
    const now = this.ctx.currentTime;
    const g = this.ambienceGain.gain;
    g.cancelScheduledValues(now);
    g.setValueAtTime(Math.max(g.value, 0.0001), now);
    g.exponentialRampToValueAtTime(0.0001, now + fadeSecs);
    src.stop(now + fadeSecs + 0.05);
  }

  // rain_up / rain_down：环境音整体 ±6dB，4 秒过渡
  nudgeAmbience(db, when) {
    this.ambienceLevel = Math.min(dbToGain(9), Math.max(dbToGain(-12), this.ambienceLevel * dbToGain(db)));
    this.ambienceGain.gain.setTargetAtTime(dbToGain(AMBIENCE_DB) * this.ambienceLevel, when, 1.3);
  }

  fadeOutAll(seconds) {
    const now = this.ctx.currentTime;
    for (const g of [this.master.gain]) {
      g.cancelScheduledValues(now);
      g.setValueAtTime(g.value, now);
      g.linearRampToValueAtTime(0, now + seconds);
    }
  }

  setVoiceVolume(v) { this.voiceVolume.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05); }
  setAmbienceVolume(v) { this.ambienceVolume.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05); }

  // ---- 定位 ---------------------------------------------------------------

  applyPosition(azimuth, distance, when, tc = TC) {
    const d = DISTANCES[distance] || DISTANCES.mid;
    const { x, z } = coords(AZIMUTHS[azimuth] ?? 0, d.r);
    const p = this.panner;
    for (const [param, value] of [[p.positionX, x], [p.positionY, 0], [p.positionZ, z]]) {
      param.cancelScheduledValues(when);
      param.setTargetAtTime(value, when, tc);
    }
    this.shelf.gain.setTargetAtTime(d.shelfDb, when, tc);
    this.distGain.gain.setTargetAtTime(dbToGain(d.gainDb), when, tc);
    this.wet.gain.setTargetAtTime(d.wet, when, tc);
    this.pos = { azimuth, distance };
  }

  // move：沿圆弧插值方位角，同时线性过渡距离。
  applyMove(azimuth, distance, duration, when) {
    const from = this.pos, fd = DISTANCES[from.distance] || DISTANCES.mid, td = DISTANCES[distance] || DISTANCES.mid;
    const a0 = AZIMUTHS[from.azimuth] ?? 0;
    const diff = arcPath(a0, AZIMUTHS[azimuth] ?? 0);
    const steps = Math.max(8, Math.round(duration * 30));
    const xs = new Float32Array(steps), zs = new Float32Array(steps);
    for (let i = 0; i < steps; i++) {
      const t = i / (steps - 1);
      const eased = t * t * (3 - 2 * t);
      const { x, z } = coords(a0 + diff * eased, fd.r + (td.r - fd.r) * eased);
      xs[i] = x; zs[i] = z;
    }
    const p = this.panner;
    for (const [param, curve] of [[p.positionX, xs], [p.positionZ, zs]]) {
      param.cancelScheduledValues(when);
      param.setValueAtTime(curve[0], when);
      param.setValueCurveAtTime(curve, when + 0.001, duration);
    }
    for (const [param, a, b] of [
      [this.shelf.gain, fd.shelfDb, td.shelfDb],
      [this.distGain.gain, dbToGain(fd.gainDb), dbToGain(td.gainDb)],
      [this.wet.gain, fd.wet, td.wet],
    ]) {
      param.cancelScheduledValues(when);
      param.setValueAtTime(a, when);
      param.linearRampToValueAtTime(b, when + duration);
    }
    this.pos = { azimuth, distance };
  }

  async playSfx(name, when, { spatial = true } = {}) {
    if (!this.sfxCache.has(name)) this.sfxCache.set(name, loadSfx(this.ctx, name));
    const buffer = await this.sfxCache.get(name);
    if (!buffer) return 0;
    const src = new AudioBufferSourceNode(this.ctx, { buffer });
    // 音效走人声链路，跟着角色位置；但绕过打断淡出，以免被截断得太突兀
    src.connect(spatial ? this.shelf : this.master);
    src.start(Math.max(when, this.ctx.currentTime));
    this.track(src);
    return buffer.duration;
  }

  // ---- 播放队列 -----------------------------------------------------------

  onSegment(msg) {
    this.segments.set(msg.seg, { ...msg, start: null, ended: false, lastEnd: 0 });
  }

  pushAudio(segId, int16) {
    const seg = this.segments.get(segId);
    if (!seg || seg.dropped || !int16.length) return;
    const ctx = this.ctx;
    if (seg.start === null) this.startSegment(seg);
    const buffer = ctx.createBuffer(1, int16.length, TTS_RATE);
    const ch = buffer.getChannelData(0);
    for (let i = 0; i < int16.length; i++) ch[i] = int16[i] / 32768;
    const src = new AudioBufferSourceNode(ctx, { buffer });
    src.connect(this.voiceIn);
    const at = Math.max(this.cursor, ctx.currentTime + 0.02); // 断流时从当前时刻续上
    src.start(at);
    this.cursor = at + buffer.duration;
    seg.lastEnd = this.cursor;
    this.track(src);
  }

  onSegmentEnd(segId) {
    const seg = this.segments.get(segId);
    if (!seg || seg.dropped) return;
    seg.ended = true;
    if (seg.start === null) this.startSegment(seg); // 没有音频（只有舞台指令，或 TTS 失败）
    const endAt = Math.max(seg.lastEnd, seg.start);
    this.cursor = Math.max(this.cursor, endAt) + (seg.kind === "filler" ? 0.05 : GAP);
    this.at(endAt, () => {
      this.segments.delete(segId);
      this.reportEnd(seg);
    });
  }

  startSegment(seg) {
    const ctx = this.ctx;
    const now = ctx.currentTime;
    let start = this.cursor < now ? now + LEAD : this.cursor;
    seg.start = start;
    // 独白逐轮衰减；普通回复回到 0dB
    this.level.gain.setTargetAtTime(dbToGain(seg.gain_db || 0), start, 0.2);
    let delay = 0;
    for (const cue of seg.cues || []) {
      if (cue.kind === "pos") this.applyPosition(cue.azimuth, cue.distance, start);
      else if (cue.kind === "move") this.applyMove(cue.azimuth, cue.distance, cue.duration || 2, start);
      else if (cue.kind === "sfx") {
        if (cue.name === "rain_up") this.nudgeAmbience(6, start);
        else if (cue.name === "rain_down") this.nudgeAmbience(-6, start);
        else {
          const sfxAt = start + delay;
          delay += 0.6;
          this.playSfx(cue.name, sfxAt).then((dur) => {
            // 纯音效段落：让后面的台词等音效放完一大半
            if (!seg.text) this.cursor = Math.max(this.cursor, sfxAt + dur * 0.7);
          });
        }
      }
    }
    if (seg.text && delay) start += delay; // 先听到音效，再开口
    this.cursor = start;
    this.at(seg.start, () => this.reportStart(seg));
  }

  // 打断：150ms 内淡出、清空队列；未播完的段落不回报 seg_end。
  interrupt() {
    const ctx = this.ctx;
    const now = ctx.currentTime;
    const g = this.fade.gain;
    g.cancelScheduledValues(now);
    g.setValueAtTime(g.value, now);
    g.linearRampToValueAtTime(0, now + FADE_SECS);
    g.setValueAtTime(1, now + FADE_SECS + 0.03);
    for (const src of this.sources) {
      try { src.stop(now + FADE_SECS + 0.02); } catch (_) { /* 已停止 */ }
    }
    for (const t of this.timers) clearTimeout(t);
    this.timers.clear();
    for (const seg of this.segments.values()) seg.dropped = true;
    this.segments.clear();
    this.cursor = now + FADE_SECS + 0.05;
  }

  get busy() {
    return this.segments.size > 0 || this.cursor > this.ctx.currentTime;
  }

  track(src) {
    this.sources.add(src);
    src.onended = () => this.sources.delete(src);
  }

  // 在音频时钟的某个时刻回调（setTimeout 精度足够用来回报进度）。
  at(when, fn) {
    const ms = Math.max(0, (when - this.ctx.currentTime) * 1000);
    const t = setTimeout(() => {
      this.timers.delete(t);
      fn();
    }, ms);
    this.timers.add(t);
  }
}
