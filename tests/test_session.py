import asyncio

import pytest

from asmr.state import Timing

from .fakes import FakeTTS, make_session, wait_for


@pytest.fixture
async def running():
    tasks = []

    async def start(**kwargs):
        session, browser, stt, llm, tts = make_session(**kwargs)
        tasks.append((asyncio.create_task(session.run()), browser))
        return session, browser, stt, llm, tts

    yield start
    for task, browser in tasks:
        await browser.inbox.put(None)
        await asyncio.wait_for(task, 2)


async def test_intro_plays_first_mes_then_waits(running):
    session, browser, *_ = await running()
    await wait_for(lambda: session.sm.state.value == "waiting")
    segs = browser.of_type("segment")
    assert segs[0]["cues"] == [{"kind": "pos", "azimuth": "left", "distance": "near"}]
    assert [s["text"] for s in segs] == ["你来啦。", "今天辛苦了。"]
    assert session.history == [{"role": "assistant", "content": "[pos: left, near]\n你来啦。\n今天辛苦了。"}]
    assert browser.states()[:2] == ["intro", "waiting"]


async def test_reply_flow_and_history(running):
    session, browser, stt, llm, tts = await running(replies=["[style: 耳语]<sigh> 辛苦了。闭上眼睛。好好睡。"])
    await wait_for(lambda: session.sm.state.value == "waiting")
    stt.texts[b"a"] = "今天好累"
    await browser.say(b"a")
    await wait_for(lambda: len(session.history) == 3)
    assert browser.of_type("transcript")[0]["text"] == "今天好累"
    assert session.history[1] == {"role": "user", "content": "今天好累"}
    assert session.history[2]["content"] == "[style: 耳语]\n<sigh> 辛苦了。\n闭上眼睛。 好好睡。"
    # LLM 看到了 system + 开场白 + 用户的话
    messages = llm.calls[0]
    assert messages[0]["role"] == "system" and "舞台指令" in messages[0]["content"]
    assert [m["role"] for m in messages[1:]] == ["assistant", "user"]
    # style 走 TTS 参数，标签不进文本
    assert tts.calls[-2] == ("<sigh> 辛苦了。", "耳语")
    assert all("[" not in text for text, _ in tts.calls)
    await wait_for(lambda: session.sm.state.value == "waiting")
    assert browser.states()[-4:] == ["listening", "thinking", "speaking", "waiting"]


async def test_barge_in_keeps_only_played_segments(running):
    session, browser, stt, llm, tts = await running(
        replies=["第一句。第二句。第三句。第四句。"], tts=FakeTTS(delay=0.01)
    )
    await wait_for(lambda: session.sm.state.value == "waiting")
    browser.autoplay = False
    stt.texts[b"a"] = "讲个故事"
    await browser.say(b"a")
    await wait_for(lambda: len(browser.of_type("segment")) >= 3)
    first = browser.of_type("segment")[2]["seg"]  # 前两段是开场白
    await browser.report("seg_start", first)
    await asyncio.sleep(0.02)
    await browser.inbox.put(b"S")  # 用户开口打断
    await wait_for(lambda: browser.of_type("interrupt"))
    await wait_for(lambda: session.sm.state.value == "listening")
    assert session.history[-2] == {"role": "user", "content": "讲个故事"}
    assert session.history[-1] == {"role": "assistant", "content": "第一句。——"}
    # 打断之后不再下发旧回复的段落
    count = len(browser.of_type("segment"))
    await asyncio.sleep(0.1)
    assert len(browser.of_type("segment")) == count


async def test_interrupted_before_anything_played_merges_user_text(running):
    session, browser, stt, llm, tts = await running(replies=["好。", "明白了。"], tts=FakeTTS(delay=0.05))
    await wait_for(lambda: session.sm.state.value == "waiting")
    browser.autoplay = False
    stt.texts[b"a"] = "我想"
    stt.texts[b"b"] = "听雨声"
    await browser.say(b"a")
    await wait_for(lambda: browser.of_type("segment")[2:])
    await browser.say(b"b")  # 还没播出任何一句就接着说
    browser.autoplay = True
    await wait_for(lambda: len(llm.calls) == 2)
    assert llm.calls[1][-1] == {"role": "user", "content": "我想 听雨声"}
    await wait_for(lambda: session.history[-1]["content"] == "明白了。")
    assert session.history[-2] == {"role": "user", "content": "我想 听雨声"}


async def test_silence_leads_to_monologue_without_questions(running):
    timing = Timing(t1=0.05, silence_to_sleep=10_000)
    session, browser, stt, llm, tts = await running(
        replies=["[style: 很轻]雨还在下呢。你还在吗？我就在这儿。"], timing=timing
    )
    await wait_for(lambda: any(s["kind"] == "monologue" for s in browser.of_type("segment")))
    mono = [s for s in browser.of_type("segment") if s["kind"] == "monologue"]
    assert mono[0]["gain_db"] == -2
    assert "你还在吗" not in "".join(s["text"] for s in mono)
    assert [s["text"] for s in mono if s["text"]] == ["雨还在下呢。 我就在这儿。"]  # 独白整段合成
    assert "不要提问" in llm.calls[0][-1]["content"]


async def test_tts_failure_skips_segment(running):
    session, browser, stt, llm, tts = await running(
        first_mes="你好。危险的一句。晚安。", tts=FakeTTS(fail_on="危险")
    )
    await wait_for(lambda: session.sm.state.value == "waiting")
    assert [s["text"] for s in browser.of_type("segment")] == ["你好。", "危险的一句。 晚安。"]
    assert len(browser.of_type("segment_end")) == 2


async def test_sleep_then_end(running):
    timing = Timing(t1=1000, silence_to_sleep=0.05, sleep_timer=0.05)
    session, browser, *_ = await running(timing=timing)
    await wait_for(lambda: browser.of_type("end"))
    assert browser.states()[-2:] == ["sleep", "ended"]
