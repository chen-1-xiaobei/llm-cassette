"""稳定指纹：把一次模型调用压成可比较的键。

键必须只包含**参与决策**的内容。LangChain 的消息对象还带 `additional_kwargs`、
`response_metadata`（token 用量、响应 ID、时间戳）等易变字段，一旦算进键，
同一段对话两次调用就会得到不同指纹，回放必然失效。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

_ROLE_BY_CLASS = {
    "SystemMessage": "system",
    "HumanMessage": "human",
    "AIMessage": "ai",
    "ToolMessage": "tool",
}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def message_to_dict(message: Any) -> dict:
    """把一条消息压成只含决策相关字段的 dict。"""
    payload: dict[str, Any] = {
        "role": _ROLE_BY_CLASS.get(type(message).__name__, type(message).__name__),
        "content": _content_to_text(getattr(message, "content", "")),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            {"name": call.get("name"), "args": call.get("args", {}), "id": call.get("id")}
            for call in tool_calls
        ]
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    return payload


def request_key(model: str, messages: Iterable[Any], tool_names: Iterable[str] = (),
                extra: str = "") -> str:
    """计算一次请求的指纹。

    参数
    ----
    model: 模型标识（参与计算，换模型即视为不同请求）
    messages: 消息列表
    tool_names: 当前绑定的工具名。**刻意参与计算**——增删工具会让旧录像失效，
        这是正确的：工具集合变了，模型的决策依据也变了。
    extra: 额外区分项（例如结构化输出时带上 schema 名），留空则不影响结果。
    """
    payload = {
        "model": model,
        "tools": sorted(tool_names),
        "messages": [message_to_dict(message) for message in messages],
    }
    if extra:
        payload["extra"] = extra
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
