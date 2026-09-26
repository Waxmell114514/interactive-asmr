from asmr.segments import Segmenter
from asmr.tags import Position, TagParser


def run(text, **kwargs):
    parser = TagParser(style="温柔")
    seg = Segmenter(position=Position(), **kwargs)
    out = []
    for event in parser.feed(text) + parser.flush():
        out += seg.push(event)
    return out + seg.flush()


def test_first_sentence_alone_then_groups_of_three():
    segs = run("一。二。三。四。五。六。")
    assert [s.text for s in segs] == ["一。", "二。 三。 四。", "五。 六。"]


def test_cues_break_groups_and_attach_to_next_segment():
    segs = run("一。二。[pos: right, near]三。[sfx: page_turn]")
    assert [s.text for s in segs] == ["一。", "二。", "三。", ""]
    assert segs[2].cues[0].azimuth == "right"
    assert segs[2].position == Position("right", "near")
    assert segs[3].cues[0].name == "page_turn"


def test_style_change_breaks_and_renders():
    segs = run("一。二。[style: 耳语]三。")
    assert segs[2].style == "耳语"
    assert segs[2].render() == "[style: 耳语]\n三。"
    assert segs[1].style == "温柔"


def test_monologue_is_one_segment():
    segs = run("一。二。三。四。五。", whole=True)
    assert [s.text for s in segs] == ["一。 二。 三。 四。 五。"]


def test_render_roundtrip_is_parseable():
    segs = run("[pos: left, near][style: 轻]嗯。[move: right, 2s]好。")
    rendered = "\n".join(s.render() for s in segs)
    again = run(rendered)
    assert [(s.text, s.style, s.position) for s in again] == [(s.text, s.style, s.position) for s in segs]
