"""指纹算法的单测：稳定性与敏感度。"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from llm_cassette import message_to_dict, request_key


def test_roles_are_normalized():
    assert message_to_dict(SystemMessage("s"))["role"] == "system"
    assert message_to_dict(HumanMessage("h"))["role"] == "human"
    assert message_to_dict(AIMessage("a"))["role"] == "ai"
    assert message_to_dict(ToolMessage(content="r", tool_call_id="c1"))["role"] == "tool"


def test_tool_calls_are_kept():
    message = AIMessage(content="", tool_calls=[
        {"name": "search", "args": {"q": "x"}, "id": "c1"},
    ])
    payload = message_to_dict(message)
    assert payload["tool_calls"] == [{"name": "search", "args": {"q": "x"}, "id": "c1"}]


def test_volatile_metadata_is_ignored():
    """token 用量、响应元数据这类字段必须排除，否则键永远对不上。"""
    plain = AIMessage(content="同样的话")
    noisy = AIMessage(content="同样的话", additional_kwargs={"token_usage": 42})
    assert request_key("m", [plain]) == request_key("m", [noisy])


def test_key_is_order_insensitive_for_tools():
    messages = [HumanMessage("q")]
    assert request_key("m", messages, {"a", "b"}) == request_key("m", messages, {"b", "a"})


def test_key_changes_when_tool_set_changes():
    messages = [HumanMessage("q")]
    assert request_key("m", messages, {"search"}) != request_key("m", messages, {"search", "table"})


def test_key_changes_when_content_or_model_changes():
    assert request_key("m", [HumanMessage("a")]) != request_key("m", [HumanMessage("b")])
    assert request_key("m1", [HumanMessage("a")]) != request_key("m2", [HumanMessage("a")])


def test_key_changes_with_extra():
    messages = [HumanMessage("q")]
    assert request_key("m", messages, extra="schema:A") != request_key("m", messages, extra="schema:B")
    assert request_key("m", messages) == request_key("m", messages, extra="")


def test_key_is_stable_across_runs():
    messages = [SystemMessage("s"), HumanMessage("q")]
    assert request_key("m", messages, {"t"}, "x") == request_key("m", messages, {"t"}, "x")
