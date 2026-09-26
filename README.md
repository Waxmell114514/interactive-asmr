# 互动 ASMR

戴耳机、以语音进行的中文角色扮演 ASMR：你说话，AI 用温和、贴耳、可定位的声音回应；你安静下来，它就慢慢自己往下说，直到你睡着。

实现依据《互动 ASMR：技术 Spec v0.1》，覆盖 M0（声音验证脚本）、M1（对话跑通）和 M2（ASMR 层）。

```
浏览器麦克风 ──16kHz PCM──▶ Silero VAD + Smart Turn v3 ──▶ SenseVoice STT ──▶ LLM + 角色卡
                                                                              │ 流式文本
浏览器 Web Audio ◀──24kHz PCM + 段落/位置事件── Gemini 3.8 TTS ◀── 标签解析 + 分句/合并
(HRTF · 近讲效应 · 小房间混响 · 环境音 · 限幅)                        会话状态机（打断 / 独白 / 入睡）
```

## 快速开始

需要 Python 3.10+，一张 8GB+ 显存的 GPU 跑 STT 更好（CPU 也能跑，只是慢一些）。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[sensevoice,dev]"      # 或 ".[whisper,dev]" 用 faster-whisper
cp .env.example .env                     # 填 LLM_API_KEY、GEMINI_API_KEY，GPU 的话 STT_DEVICE=cuda
python -m asmr.server                    # 或 asmr-server
```

用桌面版 Chrome / Edge 打开 http://127.0.0.1:8000 ，戴上耳机，选角色，点“开始”。首次启动会下载 SenseVoice 模型；应声词会在第一次会话时用当前音色合成并缓存到 `data/fillers/`。

建议先点一次“空间感盲听测试”，确认耳机左右没戴反。

## M0：先验证声音

```bash
python scripts/m0_voice_lab.py design          # Voice Design 生成 3 个候选音色，试听 out/m0/voices/*.wav
python scripts/m0_voice_lab.py synth           # 每个音色 × 5 条测试句 × 两种 style 传法，记录首包延迟
python scripts/m0_voice_lab.py synth --check   # 另外用 SenseVoice 转写，检查 style 有没有被念出来
python -m asmr.fillers                         # 选定音色后预生成应声词（可选，会话中也会自动补齐）
```

结果在 `out/m0/report.md`。选定音色后把 `voice_...` 填进 `.env` 的 `TTS_VOICE`，或者写进角色卡的 `extensions.asmr.voice`。

关于 Spec 里“style 如何传给 TTS 才不会被念出来”这个待验证项：Gemini 3.8 TTS 的文档写明 `text` 按逐字稿处理，语气放进 `speech_metadata.style` 注解。所以默认用 `TTS_STYLE_MODE=annotation`；`prefix`（把 style 写进文本前缀）只留给 M0 对比试听。`synth` 两种都会生成，可以直接对比。

## 目录

| 路径 | 内容 |
| --- | --- |
| `asmr/tags.py` | LLM 输出协议的流式解析：`[pos/move/style/sfx]` 舞台指令、`<sigh>` 等声音事件白名单、按 `。！？…～` / 40 字切句 |
| `asmr/segments.py` | 把句子组装成 TTS 段落：首句单独合成、后续 2–3 句合并、独白整段合成，舞台指令处断开 |
| `asmr/state.py` | 会话状态机（纯逻辑）：Intro / Waiting / Listening / Thinking / Speaking / Monologue / Sleep |
| `asmr/session.py` | 单场会话的编排：轮次 → STT → LLM → 解析 → TTS → 下发；打断、独白、入睡；只把实际播出的句子写进历史 |
| `asmr/turn.py` | Silero VAD + Smart Turn v3 轮次检测（用 Pipecat 附带的模型实现） |
| `asmr/tts.py` | Gemini 3.8 TTS（Interactions API，SSE 流式，`speech_metadata.style`） |
| `asmr/stt.py` | SenseVoice（FunASR）/ faster-whisper |
| `asmr/llm.py` | OpenAI 兼容的流式 LLM（DeepSeek / Gemini / Claude 等） |
| `asmr/loudness.py` | 后端慢速自动增益，把耳语和正常音量拉到统一电平 |
| `asmr/fillers.py` | 应声词缓存 |
| `asmr/character.py` | 酒馆 Character Card V2（`.json` 或 PNG 卡）加载，系统提示词 |
| `asmr/server.py` | FastAPI：静态前端、`/api/characters`、`/ws` |
| `web/audio-engine.js` | 播放队列、HRTF 定位、`move` 圆弧移动、近讲效应、混响、环境音、限幅 |
| `web/sounds.js` | 程序化生成的房间脉冲响应、雨声 / 壁炉、翻书 / 倒茶 / 布料音效（可用真实素材替换，见 `web/sounds/README.md`） |
| `web/mic-worklet.js` | AudioWorklet 麦克风采集，降采样到 16kHz |
| `characters/` | 角色卡，放 `.json` 或酒馆 `.png` 即可 |
| `scripts/m0_voice_lab.py` | M0 声音验证 |

## 角色卡

沿用酒馆 V2 格式（`name`、`description`、`personality`、`scenario`、`first_mes`、`mes_example`、`system_prompt`…），`{{char}}` / `{{user}}` 会被替换。ASMR 专用设置放在 `data.extensions.asmr`：

```json
{"voice": "voice_xxx", "style": "温柔，很轻，慢慢地", "position": "front, mid", "ambience": "rain", "user_name": "你"}
```

`first_mes` 作为开场白，可以带舞台指令；留空时让 LLM 生成开场。示例见 `characters/yuki.json`。

## 体验要求怎么落实

| 编号 | 做法 |
| --- | --- |
| X1 不抢话 | VAD 静音 0.2s 后交给 Smart Turn v3 判断；判“没说完”就继续等，静音满 3s（`TURN_STOP_SECS`）才兜底结束。转写期间用户又开口，这段话并入下一轮；回复还没播出就被接话，也会把两段合并后重新回复 |
| X2 可打断 | VAD 检测到开口（0.2s）即取消 LLM / TTS 任务并通知前端，前端 120ms 线性淡出、清空队列 |
| X3 声线一致 | 固定 Voice Design 音色；style 只放短语气词；独白整段合成、回复 2–3 句合并，减少拼接感 |
| X4 语气跟随剧情 | `[style: …]` 走 `speech_metadata.style`；`<sigh>` 等白名单标记原样保留；未知指令 / 标记丢弃并记日志，绝不进 TTS 文本 |
| X5 空间感 | 浏览器 `PannerNode(HRTF)`，near 时 lowshelf +4dB、混响 3%；页面自带盲听测试 |
| X6 沉默不追问 | 独白提示词要求“不要提问”；另外在代码里丢弃独白中的“你还在吗 / 睡着了吗”类句子 |
| X7 响度安全 | 后端慢速自动增益 + 软削波（−1dBFS）；前端 DynamicsCompressor 限幅 −3dBFS；环境音 −18dB |

会话节奏默认值与 Spec 一致：沉默 12s 转独白、每次翻倍上限 60s；连续 5 次独白无回应或累计沉默 10 分钟进入入睡；独白每轮 −2dB、句数递减；入睡 20 分钟后环境音淡出结束。都可以在 `.env` 里改。

## WebSocket 协议

下行：文本帧是 JSON 事件（`state` / `segment` / `segment_end` / `interrupt` / `transcript` / `end`），二进制帧是 `4 字节小端段号 + 24kHz PCM`。舞台事件挂在段落上，在该段开始播放的那一刻按音频时钟生效。

上行：二进制帧是 16kHz PCM；文本帧 `seg_start` / `seg_end` 回报播放进度。后端只在收到 `seg_start` 后才把这一段算作“说过”，这样被打断时历史里只有真正播出的句子（被截断的那句末尾标上“——”）。

细节见 `asmr/session.py` 顶部注释。

## 和 Spec 的差异

- **骨架**：没有用 Pipecat 的帧流水线，而是 FastAPI + WebSocket 自写编排（Spec 里的备选方案），但轮次检测直接用 Pipecat 附带的 Silero VAD 和 Smart Turn v3。原因是这个项目以浏览器的播放进度为准：音频由前端调度（要按音频时钟对齐 HRTF 位置和音效），历史只记录前端确认播出的段落，独白 / 入睡计时也跟着播放状态走。Pipecat 的输出传输在服务端按实时节奏推音频，要把这些语义塞进去，改动反而比自写编排多。
- **style 传参**：Spec 写作时还是待验证项，现在 Gemini 3.8 TTS 有了 `speech_metadata.style`，直接采用；M0 脚本仍保留与文本前缀的对比。
- **分句**：除了 `。！？…～` 和满 40 字，换行和舞台指令也作为切句点（指令要在某一句开始时生效）；满 40 字时优先在最后一个逗号处切。
- **环境音和音效**：仓库里不带录音素材，默认在浏览器里程序化生成，可以换成真实录音。

## 还没做（M3 及以后）

- M3 里的多角色卡已经支持（往 `characters/` 放卡即可）；其余还没做：SOFA 近场 HRTF、本地 CosyVoice 3 备选 TTS（目前 TTS 报错时跳过该段、会话继续）、跨会话长期记忆。
- 延迟和成本只按 Spec 估算过，没有真机实测；M0 脚本会测 TTS 首包，端到端延迟需要接上真实服务后再测。
- X2 的 300ms：VAD 起步 200ms + 网络 + 前端淡出 120ms，理论上贴着上限，可以调低 `VAD_START_SECS` 换取更快的打断（代价是更容易被杂音打断）。
- 手机锁屏后浏览器会挂起音频，v1 以桌面浏览器为准。

## 测试

```bash
pytest
```

覆盖标签解析（含逐字流式输入）、段落组装、状态机计时、Gemini SSE 解析、响度、轮次检测、整场会话编排（假 STT / LLM / TTS / 浏览器），以及 FastAPI WebSocket 往返。
