"""录像与回放的核心行为。"""
import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from _fakes import FakeChatModel, FakeTool, tool_call
from llm_cassette import (
    CASSETTE_VERSION,
    Cassette,
    CassetteMissing,
    CassetteModel,
    record_session,
    replay_session,
    wrap_model,
)


def _record(path, responses, tool_names=("search",), model_name="m"):
    cassette = Cassette(path, "record")
    model = CassetteModel(cassette, FakeChatModel(responses), model_name=model_name)
    model = model.bind_tools([FakeTool(name) for name in tool_names]) if tool_names else model
    return cassette, model


def _replay(path, tool_names=("search",), model_name="m"):
    model = CassetteModel(Cassette(path, "replay"), model_name=model_name)
    return model.bind_tools([FakeTool(name) for name in tool_names]) if tool_names else model


# ---------- 录制 -> 回放 ----------

def test_roundtrip_preserves_tool_calls(tmp_path):
    path = tmp_path / "c.json"
    _, recorder = _record(path, [tool_call("search", {"q": "x"}, "c1")])
    recorder.invoke([HumanMessage("问题")])

    replayed = _replay(path).invoke([HumanMessage("问题")])
    assert replayed.content == ""
    assert replayed.tool_calls[0]["name"] == "search"
    assert replayed.tool_calls[0]["args"] == {"q": "x"}
    assert replayed.tool_calls[0]["id"] == "c1"


def test_roundtrip_preserves_plain_text(tmp_path):
    path = tmp_path / "c.json"
    _, recorder = _record(path, [AIMessage("三十天内提交")])
    recorder.invoke([HumanMessage("q")])
    assert _replay(path).invoke([HumanMessage("q")]).content == "三十天内提交"


def test_record_mode_delegates_to_real_model(tmp_path):
    real = FakeChatModel([AIMessage("真实回答")])
    model = CassetteModel(Cassette(tmp_path / "c.json", "record"), real, model_name="m")
    assert model.invoke([HumanMessage("q")]).content == "真实回答"
    assert real.calls == 1


# ---------- 严格性（最重要的一条） ----------

def test_missing_record_raises(tmp_path):
    (tmp_path / "c.json").write_text(
        json.dumps({"version": CASSETTE_VERSION, "entries": []}), encoding="utf-8"
    )
    with pytest.raises(CassetteMissing):
        _replay(tmp_path / "c.json").invoke([HumanMessage("没录过的问题")])


def test_exhausted_record_raises(tmp_path):
    path = tmp_path / "c.json"
    _, recorder = _record(path, [AIMessage("一")])
    recorder.invoke([HumanMessage("q")])

    model = _replay(path)
    assert model.invoke([HumanMessage("q")]).content == "一"
    with pytest.raises(CassetteMissing):
        model.invoke([HumanMessage("q")])


def test_changed_prompt_raises(tmp_path):
    """提示词变一个字，指纹就变，回放必须失败——这是保护，不是麻烦。"""
    path = tmp_path / "c.json"
    _, recorder = _record(path, [AIMessage("回答")])
    recorder.invoke([HumanMessage("system v1"), HumanMessage("q")])

    with pytest.raises(CassetteMissing):
        _replay(path).invoke([HumanMessage("system v2"), HumanMessage("q")])


def test_changed_tool_set_raises(tmp_path):
    path = tmp_path / "c.json"
    _, recorder = _record(path, [AIMessage("回答")], tool_names=("search",))
    recorder.invoke([HumanMessage("q")])

    with pytest.raises(CassetteMissing):
        _replay(path, tool_names=("search", "list")).invoke([HumanMessage("q")])


def test_same_key_consumed_in_order(tmp_path):
    path = tmp_path / "c.json"
    _, recorder = _record(path, [AIMessage("第一次"), AIMessage("第二次")])
    recorder.invoke([HumanMessage("q")])
    recorder.invoke([HumanMessage("q")])

    model = _replay(path)
    assert model.invoke([HumanMessage("q")]).content == "第一次"
    assert model.invoke([HumanMessage("q")]).content == "第二次"


# ---------- 流式 ----------

def test_stream_record_and_replay(tmp_path):
    path = tmp_path / "c.json"
    _, recorder = _record(path, [["员工需在 ", "三十天内 ", "提交。"]])

    async def collect(model):
        return [piece async for piece in model.astream([HumanMessage("q")])]

    assert "".join(asyncio.run(collect(recorder))) == "员工需在 三十天内 提交。"

    replay_model = _replay(path)
    replay_model.replay_chunk = 4
    pieces = asyncio.run(collect(replay_model))
    assert "".join(pieces) == "员工需在 三十天内 提交。"
    assert len(pieces) > 1, "回放应分块产出"


# ---------- 文件格式 ----------

def test_file_is_human_readable(tmp_path):
    path = tmp_path / "c.json"
    _, recorder = _record(path, [AIMessage("中文回答")])
    recorder.invoke([HumanMessage("报销流程")])

    text = path.read_text(encoding="utf-8")
    assert "中文回答" in text and "报销流程" in text
    assert json.loads(text)["version"] == CASSETTE_VERSION


def test_version_mismatch_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 999, "entries": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        Cassette(path, "replay")


def test_replay_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Cassette(tmp_path / "missing.json", "replay")


def test_invalid_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        Cassette(tmp_path / "c.json", "whatever")


# ---------- 脱敏 ----------

def test_redact_applies_to_disk_but_not_to_replay(tmp_path):
    """脱敏只改落盘内容，不影响指纹与回放结果。"""
    path = tmp_path / "c.json"

    def redact(payload):
        text = json.dumps(payload, ensure_ascii=False).replace("13800000000", "<phone>")
        return json.loads(text)

    cassette = Cassette(path, "record", redact=redact)
    model = CassetteModel(cassette, FakeChatModel([AIMessage("已记录 13800000000")]), model_name="m")
    model.invoke([HumanMessage("我的手机号是 13800000000")])

    on_disk = path.read_text(encoding="utf-8")
    assert "13800000000" not in on_disk
    assert "<phone>" in on_disk

    # 指纹算在脱敏之前，所以还能正常回放（这里不绑定工具，与录制时一致）
    replayed = CassetteModel(Cassette(path, "replay"), model_name="m")
    assert replayed.invoke([HumanMessage("我的手机号是 13800000000")]).content


# ---------- 接入方式 ----------

def test_wrap_model_is_noop_without_env(monkeypatch):
    monkeypatch.delenv("LLM_CASSETTE", raising=False)
    real = FakeChatModel([])
    assert wrap_model(real) is real


def test_wrap_model_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CASSETTE", str(tmp_path / "c.json"))
    monkeypatch.setenv("LLM_CASSETTE_MODE", "record")
    assert isinstance(wrap_model(FakeChatModel([]), model_name="m"), CassetteModel)


def test_wrap_model_accepts_path_and_mode(tmp_path):
    wrapped = wrap_model(FakeChatModel([]), tmp_path / "c.json", mode="record", model_name="m")
    assert isinstance(wrapped, CassetteModel)


def test_sessions(tmp_path):
    path = tmp_path / "c.json"
    with record_session(path) as cassette:
        assert cassette.mode == "record"
        cassette.put("k", {"messages": []}, {"content": "x"})
    assert path.exists()

    with replay_session(path) as cassette:
        assert cassette.mode == "replay"
        assert cassette.take("k")["response"]["content"] == "x"
