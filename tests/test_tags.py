import logging

import pytest

from asmr.tags import Cue, Position, Sentence, StyleChange, TagParser, parse_all

SPEC_EXAMPLE = """[pos: left, near] [style: 耳语，很轻，带一点笑意]
嗯……今天辛苦了。<sigh> 闭上眼睛，听我说就好。
[move: right, 3s]
我在这边哦。<short pause> 感觉到了吗？
[sfx: page_turn]
"""


def texts(events):
    return [e.text for e in events if isinstance(e, Sentence)]


def test_spec_example():
    events = parse_all(SPEC_EXAMPLE)
    assert events == [
        Cue("pos", "left", "near"),
        StyleChange("耳语，很轻，带一点笑意"),
        Sentence("嗯……", "耳语，很轻，带一点笑意"),
        Sentence("今天辛苦了。", "耳语，很轻，带一点笑意"),
        Sentence("<sigh> 闭上眼睛，听我说就好。", "耳语，很轻，带一点笑意"),
        Cue("move", "right", "near", duration=3.0),
        Sentence("我在这边哦。", "耳语，很轻，带一点笑意"),
        Sentence("<short pause> 感觉到了吗？", "耳语，很轻，带一点笑意"),
        Cue("sfx", name="page_turn"),
    ]


@pytest.mark.parametrize("step", [1, 2, 3, 7])
def test_streaming_matches_one_shot(step):
    parser = TagParser()
    events = []
    for i in range(0, len(SPEC_EXAMPLE), step):
        events += parser.feed(SPEC_EXAMPLE[i : i + step])
    events += parser.flush()
    assert events == parse_all(SPEC_EXAMPLE)


def test_directives_never_reach_tts_text():
    events = parse_all("[mood: happy] [pos: nowhere] 你好。[sfx: explosion] <scream2> 晚安。【style：轻】嗯。")
    for text in texts(events):
        assert "[" not in text and "]" not in text and "mood" not in text
        assert "scream2" not in text
    assert texts(events) == ["你好。", "晚安。", "嗯。"]
    assert StyleChange("轻") in events
    assert not any(isinstance(e, Cue) for e in events)


def test_unknown_marks_are_logged(caplog):
    with caplog.at_level(logging.WARNING):
        parse_all("[foo: bar] <bark> 好。")
    assert "foo" in caplog.text and "bark" in caplog.text


def test_style_persists_across_parsers_and_is_truncated():
    parser = TagParser(style="温柔")
    events = parser.feed("你好。") + parser.flush()
    assert events == [Sentence("你好。", "温柔")]
    long_style = "非" * 30
    events = parse_all(f"[style: {long_style}]好。")
    assert events[0] == StyleChange("非" * 20)
    assert events[1].style == "非" * 20


def test_partial_position_keeps_other_component():
    parser = TagParser(position=Position("left", "near"))
    events = parser.feed("[pos: far]") + parser.flush()
    assert events == [Cue("pos", "left", "far")]
    assert parser.position == Position("left", "far")


def test_chinese_position_synonyms():
    assert parse_all("[pos: 右, 近]") == [Cue("pos", "right", "near")]


def test_forty_char_split_prefers_comma():
    text = "我们" * 12 + "，" + "慢慢地" * 8 + "。"
    sentences = texts(parse_all(text))
    assert sentences[0] == "我们" * 12 + "，"
    assert all(len(s) <= 40 for s in sentences)
    assert "".join(sentences) == text


def test_forty_char_split_without_punctuation():
    text = "啊" * 90
    sentences = texts(parse_all(text))
    assert [len(s) for s in sentences] == [40, 40, 10]


def test_terminator_runs_and_closers_stay_together():
    assert texts(parse_all("真的吗？！」嗯～～好。")) == ["真的吗？！」", "嗯～～", "好。"]


def test_directive_mid_sentence_splits():
    events = parse_all("我在这边[move: right]哦。")
    assert texts(events) == ["我在这边", "哦。"]
    assert isinstance(events[1], Cue)


def test_vocal_tags_normalized():
    events = parse_all("嗯 <Short  Pause> 好<LAUGH>。")
    assert texts(events) == ["嗯 <short pause> 好 <laugh> 。"]


def test_unclosed_bracket_dropped_at_end():
    events = parse_all("你好。[pos: left")
    assert texts(events) == ["你好。"]
    assert not any(isinstance(e, Cue) for e in events)


def test_punctuation_only_dropped_but_lone_vocal_tag_kept():
    assert texts(parse_all("……\n<sigh>\n")) == ["<sigh>"]
