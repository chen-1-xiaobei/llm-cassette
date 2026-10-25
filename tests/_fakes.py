"""测试用的假模型：不联网、不需要密钥。"""
from __future__ import annotations

from langchain_core.messages import AIMessage


class FakeTool:
    def __init__(self, name: str):
        self.name = name


class FakeStructured:
    """模拟 with_structured_output 返回的链。"""

    def __init__(self, schema, payload):
        self.schema = schema
        self.payload = payload

    def invoke(self, messages, **kwargs):
        if isinstance(self.payload, dict) and isinstance(self.schema, type):
            return self.schema(**self.payload)
        return self.payload


class FakeChatModel:
    """按顺序吐出预设响应。

    responses 的每个元素可以是：
    - AIMessage：普通回复 / 工具调用
    - list[str]：流式分块
    - dict：结构化输出（配合 with_structured_output 使用）
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.bound_tools: list[str] | None = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [getattr(tool, "name", str(tool)) for tool in tools]
        return self

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return self._responses.pop(0)

    async def astream(self, messages, **kwargs):
        for piece in self._responses.pop(0):
            yield piece

    def with_structured_output(self, schema=None, **kwargs):
        self.calls += 1
        return FakeStructured(schema, self._responses.pop(0))


def tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])
