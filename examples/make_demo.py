"""生成两份演示录像，用来说明 `llm-cassette diff` 能看出什么。

场景：**同一个模型标识、同一段请求**，但在服务端升级之后返回变了——
模型把检索词从完整句子退化成了两个词。这类退化在回归测试里只会显示"红了"，
`diff` 才能告诉你"它把查询写短了"。

    python examples/make_demo.py
    llm-cassette diff examples/cassettes/before.json examples/cassettes/after.json
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from llm_cassette import Cassette, CassetteModel  # noqa: E402

OUT = HERE / "cassettes"
MODEL = "deepseek-chat"
QUESTION = "出差住宿一晚最多能报多少钱？"


class ScriptedModel:
    """吐一条固定的工具调用，冒充真实模型。"""

    def __init__(self, query: str):
        self.query = query

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "search_knowledge_base",
                "args": {"question": self.query, "top_k": 4},
                "id": "c1",
            }],
        )


def record(filename: str, query: str) -> None:
    path = OUT / filename
    path.unlink(missing_ok=True)
    cassette = Cassette(path, "record")
    model = CassetteModel(cassette, ScriptedModel(query), model_name=MODEL)
    model.bind_tools([type("T", (), {"name": "search_knowledge_base"})()]).invoke(
        [HumanMessage(QUESTION)]
    )


if __name__ == "__main__":
    record("before.json", "出差住宿报销标准是多少")
    record("after.json", "住宿标准")
    print(f"已生成 {OUT / 'before.json'} 和 {OUT / 'after.json'}")
