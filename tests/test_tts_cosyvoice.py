import httpx
import numpy as np
import pytest

from asmr.tts import TTSError, synthesize_all
from asmr.tts_cosyvoice import CosyVoiceConfig, CosyVoiceTTS, prepare_text


@pytest.fixture
def config(tmp_path):
    (tmp_path / "whisper.wav").write_bytes(b"RIFF-whisper")
    (tmp_path / "normal.wav").write_bytes(b"RIFF-normal")
    return {
        "instructions": [{"match": ["极轻"], "instruction": "Please say a sentence in a very soft voice."}],
        "refs": [
            {"match": ["耳语"], "audio": str(tmp_path / "whisper.wav"), "text": "悄悄告诉你。"},
            {"audio": str(tmp_path / "normal.wav"), "text": "你好呀。", "default": True},
        ],
    }


def make_tts(config, handler, **overrides):
    cfg = CosyVoiceConfig.from_dict({**config, **overrides})
    return CosyVoiceTTS(cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def parse_form(request: httpx.Request) -> dict:
    """从 multipart 请求体里取出表单字段和上传的文件内容（测试够用的简易解析）。"""
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    fields = {}
    for part in request.content.split(b"--" + boundary):
        head, _, body = part.partition(b"\r\n\r\n")
        if b'name="' not in head:
            continue
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        fields[name] = body.rstrip(b"\r\n")
    return fields


def test_prepare_text_maps_vocal_tags():
    assert prepare_text("嗯……<sigh> 今天辛苦了。<chuckle>") == "嗯……[sigh] 今天辛苦了。[laughter]"
    assert prepare_text("<short pause> 好 <whispers>") == "好"
    assert prepare_text("<breath>") == "[breath]"
    assert prepare_text("……<long pause>") == ""


def test_style_becomes_instruction_for_v3(config):
    cfg = CosyVoiceConfig.from_dict(config)
    assert cfg.instruct_text("耳语，很轻") == "You are a helpful assistant. 请用耳语，很轻的语气说这句话。<|endofprompt|>"
    assert cfg.instruct_text("极轻的耳语，很慢") == (
        "You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>"
    )
    v2 = CosyVoiceConfig.from_dict({**config, "version": 2})
    assert v2.instruct_text("温柔") == "请用温柔的语气说这句话。<|endofprompt|>"
    assert v2.prompt_text(v2.refs[1]) == "你好呀。"
    assert cfg.prompt_text(cfg.refs[1]) == "You are a helpful assistant.<|endofprompt|>你好呀。"


async def test_style_uses_instruct2_with_matching_ref(config):
    seen = {}

    def handler(request: httpx.Request):
        seen["path"] = request.url.path
        seen["form"] = parse_form(request)
        pcm = (np.arange(4800) % 100).astype(np.int16).tobytes()
        return httpx.Response(200, content=pcm)

    tts = make_tts(config, handler)
    out = await synthesize_all(tts, "闭上眼睛。<sigh>", "耳语，很轻")
    assert len(out) == 4800 * 2  # 24kHz 原样透传
    assert seen["path"] == "/inference_instruct2"
    assert seen["form"]["tts_text"].decode() == "闭上眼睛。[sigh]"
    assert "耳语，很轻" in seen["form"]["instruct_text"].decode()
    assert seen["form"]["prompt_wav"] == b"RIFF-whisper"


async def test_empty_style_uses_zero_shot(config):
    seen = {}

    def handler(request: httpx.Request):
        seen["path"] = request.url.path
        seen["form"] = parse_form(request)
        return httpx.Response(200, content=b"\x00\x00" * 100)

    await synthesize_all(make_tts(config, handler), "你好。", "")
    assert seen["path"] == "/inference_zero_shot"
    assert seen["form"]["prompt_text"].decode().endswith("<|endofprompt|>你好呀。")
    assert seen["form"]["prompt_wav"] == b"RIFF-normal"


async def test_resamples_22k_to_24k(config):
    pcm = np.zeros(22050, dtype=np.int16).tobytes()
    tts = make_tts(config, lambda r: httpx.Response(200, content=pcm), sample_rate=22050)
    out = await synthesize_all(tts, "你好。", "")
    assert abs(len(out) / 2 - 24000) < 50


async def test_errors(config, tmp_path):
    tts = make_tts(config, lambda r: httpx.Response(500, text="CUDA out of memory"))
    with pytest.raises(TTSError, match="out of memory"):
        await synthesize_all(tts, "你好。")
    tts = make_tts(config, lambda r: httpx.Response(200, content=b""))
    with pytest.raises(TTSError, match="没有返回音频"):
        await synthesize_all(tts, "你好。")
    missing = {**config, "refs": [{"audio": str(tmp_path / "nope.wav"), "text": "x"}]}
    tts = make_tts(missing, lambda r: httpx.Response(200, content=b"\x00\x00"))
    with pytest.raises(TTSError, match="找不到参考音频"):
        await synthesize_all(tts, "你好。")
