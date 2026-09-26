"""M0 声音验证：设计候选音色、批量合成测试句、对比 style 传参方式、实测首包延迟。

用法（需要 .env 里的 GEMINI_API_KEY）：

    # 1. 用 Voice Design 生成 3 个候选音色（每个会存一段官方试听 wav）
    python scripts/m0_voice_lab.py design

    # 2. 每个音色 × 测试句 × style 传法（annotation / prefix）批量合成，记录首包延迟
    python scripts/m0_voice_lab.py synth                 # 用 design 生成的音色
    python scripts/m0_voice_lab.py synth --voice Achernar --voice voice_xxx --model gemini-3.8-flash-tts

    # 3. （可选）用本地 SenseVoice 转写合成结果，自动检查 style 描述有没有被念出来
    python scripts/m0_voice_lab.py synth --check

输出都在 out/m0/：voices/*.wav、synth/*.wav、report.md。退出标准：选定 1 个音色；style 不被朗读；耳语效果满意。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asmr.config import ROOT, Settings  # noqa: E402
from asmr.tts import SAMPLE_RATE, GeminiTTS  # noqa: E402

OUT = ROOT / "out" / "m0"

CANDIDATES = [
    {
        "display_name": "asmr-姐姐-温柔",
        "gender": "female",
        "input": "一位二十多岁的中国女性，声音温柔、柔软、略带气声，语速很慢，像在深夜贴着耳边轻声说话，普通话标准，没有口音。",
    },
    {
        "display_name": "asmr-少年-低沉",
        "gender": "male",
        "input": "一位二十多岁的中国男性，声音低沉温暖、带一点沙哑，说话轻而慢，像睡前在耳边低语，普通话标准。",
    },
    {
        "display_name": "asmr-少女-清澈",
        "gender": "female",
        "input": "一位年轻的中国女性，声音清澈甜美、轻柔，带一点笑意，说话慢条斯理，适合哄人入睡的耳语，普通话标准。",
    },
]

# (台词, style)：覆盖耳语、轻笑、叹气、停顿、长句
TEST_LINES = [
    ("嗯……今天辛苦了。<sigh> 闭上眼睛，听我说就好。", "耳语，很轻，带一点笑意"),
    ("我在这边哦。<short pause> 感觉到了吗？", "耳语，贴着耳朵，很慢"),
    ("<chuckle> 你呀，总是这样逞强。", "轻笑，温柔"),
    ("外面的雨一直在下，屋檐上的水滴一颗一颗落下来，像在数着拍子，你就跟着它慢慢呼吸吧。", "很轻，很慢，像在讲睡前故事"),
    ("晚安。<long pause> 我会一直在这里的。", "极轻的耳语，几乎只有气声"),
]
STYLE_MODES = ["annotation", "prefix"]


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


async def design(args, settings: Settings) -> None:
    url = f"{settings.gemini_base_url.rstrip('/')}/voices"
    headers = {"x-goog-api-key": settings.gemini_api_key}
    results = []
    async with httpx.AsyncClient(timeout=120) as client:
        for cand in CANDIDATES:
            body = {
                "store": True,
                "voice": {
                    "model": args.model,
                    "type": "prompted",
                    "display_name": cand["display_name"],
                    "gender": cand["gender"],
                    "language_code": args.language_code,
                    "prompted": {"input": cand["input"]},
                },
            }
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                print(f"✗ {cand['display_name']}: HTTP {resp.status_code} {resp.text[:300]}")
                continue
            data = resp.json()
            voice_id = data.get("id") or data.get("name", "").split("/")[-1]
            sample = (data.get("sample_audio") or {}).get("data")
            if sample:
                path = OUT / "voices" / f"{cand['display_name']}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(sample))  # 官方返回的就是完整 wav
            print(f"✓ {cand['display_name']}: {voice_id}")
            results.append({**cand, "id": voice_id})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "voices.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"音色列表已写入 {OUT / 'voices.json'}；试听 {OUT / 'voices'}/*.wav")


async def timed_synth(tts: GeminiTTS, text: str, style: str) -> tuple[bytes, float, float]:
    start = time.perf_counter()
    first = None
    chunks = []
    async for chunk in tts.synthesize(text, style):
        if first is None:
            first = time.perf_counter() - start
        chunks.append(chunk)
    return b"".join(chunks), first or float("nan"), time.perf_counter() - start


def load_stt():
    try:
        from asmr.stt import SenseVoiceSTT

        return SenseVoiceSTT(device="cpu")
    except Exception as e:  # noqa: BLE001
        print(f"（跳过自动检查：无法加载 SenseVoice：{e}）")
        return None


def style_leak(style: str, transcript: str) -> float:
    """style 描述里的字有多少出现在转写结果里（粗略指标，越高越可能被念出来）。"""
    chars = {c for c in style if "一" <= c <= "鿿"}
    if not chars:
        return 0.0
    return len([c for c in chars if c in transcript]) / len(chars)


async def synth(args, settings: Settings) -> None:
    voices = args.voice
    if not voices:
        path = OUT / "voices.json"
        if not path.exists():
            sys.exit("没有 --voice，也没有 out/m0/voices.json；先运行 design 或指定 --voice")
        voices = [v["id"] for v in json.loads(path.read_text(encoding="utf-8"))]
    stt = load_stt() if args.check else None
    rows = []
    for model in args.model:
        for voice in voices:
            for mode in STYLE_MODES:
                tts = GeminiTTS(
                    api_key=settings.gemini_api_key,
                    voice=voice,
                    model=model,
                    style_mode=mode,
                    base_url=settings.gemini_base_url,
                )
                for i, (text, style) in enumerate(TEST_LINES):
                    try:
                        pcm, ttfb, total = await timed_synth(tts, text, style)
                    except Exception as e:  # noqa: BLE001
                        print(f"✗ {model} {voice} {mode} #{i}: {e}")
                        rows.append({"model": model, "voice": voice, "mode": mode, "line": i, "error": str(e)})
                        continue
                    name = f"{model}_{voice}_{mode}_{i}.wav".replace("/", "_")
                    write_wav(OUT / "synth" / name, pcm)
                    row = {
                        "model": model, "voice": voice, "mode": mode, "line": i, "file": name,
                        "ttfb": ttfb, "total": total, "audio_secs": len(pcm) / 2 / SAMPLE_RATE,
                    }
                    if stt is not None:
                        import numpy as np
                        import soxr

                        pcm16k = soxr.resample(np.frombuffer(pcm, dtype=np.int16), SAMPLE_RATE, 16000)
                        transcript = await stt.transcribe(pcm16k.astype(np.int16).tobytes())
                        row["transcript"] = transcript
                        row["leak"] = style_leak(style, transcript)
                    rows.append(row)
                    print(f"✓ {name}  首包 {ttfb * 1000:.0f}ms  总计 {total:.2f}s")
                await tts.aclose()
    write_report(rows)


def write_report(rows: list[dict]) -> None:
    ok = [r for r in rows if "error" not in r]
    lines = ["# M0 合成报告", "", "## 首包延迟（ms）", "", "| 模型 | 音色 | style 传法 | 中位数 | 最大 |", "|---|---|---|---|---|"]
    groups: dict[tuple, list[dict]] = {}
    for r in ok:
        groups.setdefault((r["model"], r["voice"], r["mode"]), []).append(r)
    for (model, voice, mode), rs in groups.items():
        ttfbs = [r["ttfb"] * 1000 for r in rs]
        lines.append(f"| {model} | {voice} | {mode} | {statistics.median(ttfbs):.0f} | {max(ttfbs):.0f} |")
    if any("transcript" in r for r in ok):
        lines += ["", "## style 是否被念出来（SenseVoice 转写）", "", "| 文件 | 泄漏度 | 转写 |", "|---|---|---|"]
        for r in ok:
            lines.append(f"| {r['file']} | {r['leak']:.0%} | {r['transcript']} |")
    errors = [r for r in rows if "error" in r]
    if errors:
        lines += ["", "## 失败", ""] + [f"- {r['model']} {r['voice']} {r['mode']} #{r['line']}: {r['error']}" for r in errors]
    lines += ["", "## 测试句", ""] + [f"{i}. `{t}` — style: {s}" for i, (t, s) in enumerate(TEST_LINES)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告：{OUT / 'report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("design", help="用 Voice Design 生成候选音色")
    p.add_argument("--model", default="gemini-3.8-flash-tts")
    p.add_argument("--language-code", default="cmn-CN")
    p = sub.add_parser("synth", help="批量合成测试句并记录首包延迟")
    p.add_argument("--voice", action="append", help="音色 ID 或预置音色名，可重复")
    p.add_argument("--model", action="append", help="可重复；默认 gemini-3.8-flash-lite-tts")
    p.add_argument("--check", action="store_true", help="用本地 SenseVoice 检查 style 是否被念出来")
    args = parser.parse_args()
    settings = Settings.from_env()
    if not settings.gemini_api_key:
        sys.exit("请在 .env 里设置 GEMINI_API_KEY")
    if args.cmd == "design":
        asyncio.run(design(args, settings))
    else:
        args.model = args.model or ["gemini-3.8-flash-lite-tts"]
        asyncio.run(synth(args, settings))


if __name__ == "__main__":
    main()
