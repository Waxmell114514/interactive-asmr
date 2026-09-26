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
