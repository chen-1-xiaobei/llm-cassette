"""结构化输出（with_structured_output）的录制与回放。"""
import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from _fakes import FakeChatModel
from llm_cassette import Cassette, CassetteMissing, CassetteModel


class Answer(BaseModel):
    text: str
    citations: list[int] = []
    ok: bool = True


def _recorder(path, responses):
    return CassetteModel(Cassette(path, "record"), FakeChatModel(responses), model_name="m")


def _replayer(path):
    return CassetteModel(Cassette(path, "replay"), model_name="m")


def test_structured_roundtrip_returns_model_instance(tmp_path):
    path = tmp_path / "c.json"
    recorder = _recorder(path, [{"text": "三十天内", "citations": [1, 2], "ok": False}])
    original = recorder.with_structured_output(Answer).invoke([HumanMessage("q")])
    assert isinstance(original, Answer) and original.citations == [1, 2]

    replayed = _replayer(path).with_structured_output(Answer).invoke([HumanMessage("q")])
    assert isinstance(replayed, Answer), "回放也必须返回 pydantic 实例，而不是 dict"
    assert replayed.text == "三十天内"
    assert replayed.citations == [1, 2]
    assert replayed.ok is False


def test_structured_schema_is_part_of_key(tmp_path):
    """换了 schema 就应当查不到——解析目标变了，旧记录不能代表新行为。"""

    class Other(BaseModel):
        value: str

    path = tmp_path / "c.json"
    _recorder(path, [{"text": "x"}]).with_structured_output(Answer).invoke([HumanMessage("q")])

    with pytest.raises(CassetteMissing):
        _replayer(path).with_structured_output(Other).invoke([HumanMessage("q")])


def test_structured_missing_raises(tmp_path):
    path = tmp_path / "c.json"
    recorder = _recorder(path, [{"text": "x"}])
    recorder.with_structured_output(Answer).invoke([HumanMessage("q")])

    with pytest.raises(CassetteMissing):
        _replayer(path).with_structured_output(Answer).invoke([HumanMessage("别的问题")])


def test_structured_streaming_is_explicitly_unsupported(tmp_path):
    path = tmp_path / "c.json"
    _recorder(path, [{"text": "x"}]).with_structured_output(Answer).invoke([HumanMessage("q")])

    model = _replayer(path).with_structured_output(Answer)
    with pytest.raises(NotImplementedError):
        model.astream([HumanMessage("q")])
