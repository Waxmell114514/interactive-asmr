import json
import struct

from fastapi.testclient import TestClient

from asmr.config import ROOT, Settings
from asmr.server import create_app
from asmr.session import Services

from .fakes import FakeLLM, FakeSTT, FakeTTS, FakeTurnDetector


def make_client():
    settings = Settings(intro_delay=0.0, fillers=False)
    factory = lambda character: Services(  # noqa: E731
        stt=FakeSTT(), llm=FakeLLM(), tts=FakeTTS(), turn_detector=FakeTurnDetector()
    )
    return TestClient(create_app(settings, services_factory=factory))


def test_characters_and_static_page():
    client = make_client()
    data = client.get("/api/characters").json()
    assert data["characters"][0]["id"] == "yuki"
    assert "page_turn" in data["sfx"]
    assert client.get("/").status_code == 200


def test_websocket_intro_roundtrip():
    client = make_client()
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "start", "character": "yuki"}))
        assert json.loads(ws.receive_text()) == {"type": "state", "state": "intro"}
        seg = json.loads(ws.receive_text())
        assert seg["type"] == "segment" and seg["kind"] == "intro"
        assert seg["cues"][0] == {"kind": "pos", "azimuth": "front", "distance": "mid"}
        frame = ws.receive_bytes()
        assert struct.unpack("<I", frame[:4])[0] == seg["seg"]
        ws.send_text(json.dumps({"type": "stop"}))


def test_default_character_card_loads():
    from asmr.character import load_characters

    yuki = load_characters(ROOT / "characters")["yuki"]
    assert yuki.name == "小雪"
    prompt = yuki.build_system_prompt(("page_turn",))
    assert "{{char}}" not in prompt and "小雪" in prompt and "page_turn" in prompt


def test_png_character_card(tmp_path):
    import base64
    import json as _json
    import struct as _struct
    import zlib

    from asmr.character import load_characters

    def chunk(kind, body):
        return _struct.pack(">I", len(body)) + kind + body + _struct.pack(">I", zlib.crc32(kind + body))

    card = {"spec": "chara_card_v2", "data": {"name": "阿澈", "first_mes": "嗨。", "extensions": {"asmr": {"position": "right, near"}}}}
    payload = b"chara\x00" + base64.b64encode(_json.dumps(card).encode())
    ihdr = _struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"tEXt", payload) + chunk(b"IEND", b"")
    (tmp_path / "che.png").write_bytes(png)
    che = load_characters(tmp_path)["che"]
    assert che.name == "阿澈" and che.position.azimuth == "right" and che.position.distance == "near"


def test_create_tts_picks_backend(tmp_path):
    import json as _json

    import pytest

    from asmr.character import Character
    from asmr.server import create_tts
    from asmr.tts import GeminiTTS
    from asmr.tts_gptsovits import GPTSoVITSTTS

    plain = Character(id="a", name="甲")
    assert isinstance(create_tts(Settings(), plain), GeminiTTS)

    settings = Settings(tts_backend="gptsovits", gptsovits_url="http://gpu:9880")
    with pytest.raises(RuntimeError, match="gptsovits"):
        create_tts(settings, plain)

    example = _json.loads((ROOT / "gptsovits.example.json").read_text(encoding="utf-8"))
    del example["url"]
    carded = Character(id="b", name="乙", gptsovits=example)
    tts = create_tts(settings, carded)
    assert isinstance(tts, GPTSoVITSTTS) and tts.config.url == "http://gpu:9880"

    path = tmp_path / "g.json"
    path.write_text(_json.dumps(example), encoding="utf-8")
    settings.gptsovits_config = path
    assert create_tts(settings, plain).config.pick_ref("耳语").audio.endswith("whisper.wav")


def test_create_tts_cosyvoice(tmp_path):
    import json as _json

    import pytest

    from asmr.character import Character
    from asmr.server import create_tts
    from asmr.tts_cosyvoice import CosyVoiceTTS

    settings = Settings(tts_backend="cosyvoice", cosyvoice_url="http://gpu:50000")
    with pytest.raises(RuntimeError, match="cosyvoice"):
        create_tts(settings, Character(id="a", name="甲"))
    example = _json.loads((ROOT / "cosyvoice.example.json").read_text(encoding="utf-8"))
    del example["url"]
    tts = create_tts(settings, Character(id="b", name="乙", cosyvoice=example))
    assert isinstance(tts, CosyVoiceTTS) and tts.config.url == "http://gpu:50000"
    assert tts.cache_key.startswith("cosyvoice|")
