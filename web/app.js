import { AudioEngine } from "./audio-engine.js";
import { setAvailableSounds } from "./sounds.js";

const $ = (id) => document.getElementById(id);

const STATE_LABELS = {
  intro: "开场",
  waiting: "在你身边",
  listening: "在听",
  thinking: "想一想",
  speaking: "在说话",
  monologue: "轻声自语",
  sleep: "入睡中",
  ended: "晚安",
};

const prefs = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(`asmr.${key}`)) ?? fallback; } catch (_) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`asmr.${key}`, JSON.stringify(value)); } catch (_) { /* 无痕模式等 */ }
  },
};

let engine = null;
let ws = null;
let mic = null;
let characters = [];

function getEngine() {
  engine ??= new AudioEngine({
    reportStart: (seg) => {
      send({ type: "seg_start", seg: seg.seg });
      if (seg.text && seg.kind !== "filler") addLine("ai", seg.text);
    },
    reportEnd: (seg) => send({ type: "seg_end", seg: seg.seg }),
  });
  return engine;
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

// ---- 角色 ------------------------------------------------------------------

async function loadCharacters() {
  const res = await fetch("api/characters");
  const data = await res.json();
  characters = data.characters;
  setAvailableSounds(data.sounds || []);
  const select = $("character");
  select.innerHTML = "";
  for (const c of characters) select.add(new Option(c.name, c.id));
  select.value = prefs.get("character", characters[0]?.id);
  onCharacterChange();
}

function onCharacterChange() {
  const c = characters.find((x) => x.id === $("character").value) || characters[0];
  if (!c) return;
  $("scenario").textContent = c.scenario || "";
  $("ambience").value = prefs.get(`ambience.${c.id}`, c.ambience || "rain");
}

// ---- 麦克风 ----------------------------------------------------------------

async function startMic(ctx) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true, // 外放调试时需要；戴耳机时回声很小
      noiseSuppression: false, // 降噪容易把耳语当噪声吃掉
      autoGainControl: true,
    },
  });
  await ctx.audioWorklet.addModule("mic-worklet.js");
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "mic-capture", { processorOptions: { targetRate: 16000 } });
  const sink = new GainNode(ctx, { gain: 0 }); // 让 worklet 保持被拉取，但不出声
  source.connect(node).connect(sink).connect(ctx.destination);
  node.port.onmessage = ({ data }) => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(data.pcm);
    $("mic-level").style.width = `${Math.min(100, Math.sqrt(data.peak) * 140)}%`;
  };
  return { stream, node, source };
}

function stopMic() {
  if (!mic) return;
  mic.node.port.onmessage = null;
  mic.source.disconnect();
  mic.stream.getTracks().forEach((t) => t.stop());
  mic = null;
  $("mic-level").style.width = "0";
}

// ---- 会话 ------------------------------------------------------------------

async function start() {
  $("error").hidden = true;
  $("start").disabled = true;
  try {
    const eng = getEngine();
    await eng.init();
    await eng.ctx.resume();
    eng.master.gain.cancelScheduledValues(eng.ctx.currentTime);
    eng.master.gain.setValueAtTime(1, eng.ctx.currentTime);
    eng.setVoiceVolume(Number($("voice-volume").value));
    eng.setAmbienceVolume(Number($("ambience-volume").value));
    mic = await startMic(eng.ctx);
    const character = $("character").value;
    prefs.set("character", character);
    prefs.set(`ambience.${character}`, $("ambience").value);
    await eng.startAmbience($("ambience").value);
    connect(character);
    $("setup").hidden = true;
    $("live").hidden = false;
    $("subtitles").innerHTML = "";
    setState("intro");
  } catch (e) {
    stopMic();
    showError(e.name === "NotAllowedError" ? "需要麦克风权限才能对话。" : `启动失败：${e.message || e}`);
  } finally {
    $("start").disabled = false;
  }
}

function connect(character) {
  const url = new URL("ws", location.href);
  url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => send({ type: "start", character });
  ws.onmessage = onMessage;
  ws.onclose = () => {
    ws = null;
    stopMic();
    if ($("orb").dataset.state !== "ended") {
      engine?.stopAmbience(3);
      setState("ended");
    }
    setTimeout(backToSetup, 4000);
  };
}

function onMessage(ev) {
  const eng = engine;
  if (ev.data instanceof ArrayBuffer) {
    const seg = new DataView(ev.data).getUint32(0, true);
    eng.pushAudio(seg, new Int16Array(ev.data, 4));
    return;
  }
  const msg = JSON.parse(ev.data);
  switch (msg.type) {
    case "state": setState(msg.state); break;
    case "segment": eng.onSegment(msg); break;
    case "segment_end": eng.onSegmentEnd(msg.seg); break;
    case "interrupt": eng.interrupt(); break;
    case "transcript": addLine("user", msg.text); break;
    case "end": eng.fadeOutAll(msg.fade_secs || 60); break;
    case "error": showError(msg.message); break;
  }
}

function stop() {
  send({ type: "stop" });
  if (engine) {
    engine.interrupt();
    engine.stopAmbience(2);
  }
  stopMic();
  ws?.close();
}

function backToSetup() {
  if (ws) return;
  $("live").hidden = true;
  $("setup").hidden = false;
}

function setState(state) {
  $("orb").dataset.state = state;
  $("state-label").textContent = STATE_LABELS[state] || state;
}

function addLine(who, text) {
  const box = $("subtitles");
  const p = document.createElement("p");
  p.className = who;
  p.textContent = text;
  box.append(p);
  while (box.children.length > 4) box.firstChild.remove();
}

function applySubtitlePref() {
  const on = $("show-subtitles").checked;
  $("subtitles").style.display = on ? "" : "none";
  prefs.set("subtitles", on);
}

function showError(text) {
  $("error").textContent = text;
  $("error").hidden = false;
  $("setup").hidden = false;
  $("live").hidden = true;
}

// ---- 空间感盲听测试（X5） --------------------------------------------------

const test = { trials: [], current: null };

async function testPlay() {
  const eng = getEngine();
  await eng.init();
  await eng.ctx.resume();
  eng.master.gain.setValueAtTime(1, eng.ctx.currentTime);
  const options = [...document.querySelectorAll("#test-guesses button")].map((b) => b.dataset.guess);
  test.current ??= options[Math.floor(Math.random() * options.length)];
  const [azimuth, distance] = test.current.split(",");
  const now = eng.ctx.currentTime;
  eng.applyPosition(azimuth, distance, now, 0.001);
  eng.playSfx("brush", now + 0.05);
}

function testGuess(guess) {
  if (!test.current) return;
  test.trials.push([test.current, guess]);
  test.current = null;
  const right = test.trials.filter(([a, b]) => a === b).length;
  const done = test.trials.length >= 8;
  $("test-result").textContent = done
    ? `完成：${right}/8 正确。${right >= 7 ? "空间感合格。" : "可以检查耳机左右是否戴反，或调小环境音再试。"}`
    : `第 ${test.trials.length} 次：${test.trials.at(-1)[0] === guess ? "对了" : "不对"}（累计 ${right}/${test.trials.length}）`;
  if (done) test.trials = [];
}

// ---- 初始化 ----------------------------------------------------------------

$("character").addEventListener("change", onCharacterChange);
$("start").addEventListener("click", start);
$("stop").addEventListener("click", stop);
$("voice-volume").addEventListener("input", (e) => engine?.setVoiceVolume(Number(e.target.value)));
$("ambience-volume").addEventListener("input", (e) => engine?.setAmbienceVolume(Number(e.target.value)));
$("show-subtitles").checked = prefs.get("subtitles", false);
$("show-subtitles").addEventListener("change", applySubtitlePref);
$("open-test").addEventListener("click", () => { $("setup").hidden = true; $("test").hidden = false; });
$("test-close").addEventListener("click", () => { $("test").hidden = true; $("setup").hidden = false; });
$("test-play").addEventListener("click", testPlay);
for (const b of document.querySelectorAll("#test-guesses button")) b.addEventListener("click", () => testGuess(b.dataset.guess));
applySubtitlePref();
loadCharacters().catch((e) => showError(`无法连接后端：${e.message}`));
