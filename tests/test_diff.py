"""两份录像的对比。"""
from langchain_core.messages import AIMessage, HumanMessage

from _fakes import FakeChatModel, tool_call
from llm_cassette import Cassette, CassetteModel, diff_files, render


def _record(path, responses, question="q", model_name="m"):
    cassette = Cassette(path, "record")
    model = CassetteModel(cassette, FakeChatModel(responses), model_name=model_name)
    model.invoke([HumanMessage(question)])
    return cassette


def test_identical_cassettes_have_no_change(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _record(a, [tool_call("search", {"q": "x"})])
    _record(b, [tool_call("search", {"q": "x"})])

    diff = diff_files(a, b)
    assert not diff.has_change
    assert diff.summary().startswith("未变 1 条")
    assert "行为一致" in render(diff)


def test_changed_tool_args_are_reported(tmp_path):
    """换模型/改提示词后，模型改了查询词——这是最需要被看见的差异。"""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _record(a, [tool_call("search", {"q": "出差住宿报销标准是多少"})])
    _record(b, [tool_call("search", {"q": "住宿标准"})])

    diff = diff_files(a, b)
    assert diff.has_change and len(diff.changed) == 1

    text = render(diff)
    assert "出差住宿报销标准是多少" in text
    assert "住宿标准" in text
    assert "改前" in text and "改后" in text


def test_changed_text_is_reported(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _record(a, [AIMessage("三十天内提交")])
    _record(b, [AIMessage("六十天内提交")])
    text = render(diff_files(a, b))
    assert "三十天内" in text and "六十天内" in text


def test_only_old_and_only_new(tmp_path):
    """输入变了（例如提示词改了）时会表现为：旧键消失、新键出现。"""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _record(a, [AIMessage("旧回答")], question="旧提示词下的问题")
    _record(b, [AIMessage("新回答")], question="新提示词下的问题")

    diff = diff_files(a, b)
    assert len(diff.only_old) == 1 and len(diff.only_new) == 1
    text = render(diff)
    assert "只在旧录像里" in text and "只在新录像里" in text


def test_diff_detects_model_switch(tmp_path):
    """换了模型名：指纹全变，两份录像对不上，必须显式告知而不是默默相等。"""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _record(a, [AIMessage("回答")], model_name="model-a")
    _record(b, [AIMessage("回答")], model_name="model-b")

    diff = diff_files(a, b)
    assert diff.only_old and diff.only_new
