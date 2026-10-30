"""录像本体与模型包装。

设计要点（三条，每条都对应一次真实的踩坑）：

1. **查不到就报错**。回放时如果没有对应记录，抛 `CassetteMissing` 而不是返回空。
   "测试全绿但其实什么都没测"比测试失败危险得多。
2. **同一输入可以有多条记录**。模型对相同输入给出不同输出是正常的，存成列表、
   按到达顺序消费；用完了再取就报错。
3. **工具集合与 schema 参与指纹**。契约变了，旧录像自动失效；不是麻烦，是保护。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from threading import Lock
from typing import Any, AsyncIterator, Callable, Iterable

from langchain_core.messages import AIMessage, AIMessageChunk

from .keys import message_to_dict, request_key

logger = logging.getLogger("agent_cassette")

CASSETTE_VERSION = 1
ENV_PATH = "AGENT_CASSETTE"
ENV_MODE = "AGENT_CASSETTE_MODE"
MODES = ("record", "replay")


def _langchain_core_version() -> str:
    try:
        return _pkg_version("langchain-core")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


class CassetteMissing(LookupError):
    """回放时找不到对应记录。故意用异常而不是返回空，避免静默通过。"""


def _content_text(chunk: Any) -> str:
    """从一个流式块里取正文文本（可能是 str，也可能是内容块列表）。"""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _assemble_tool_calls(chunks: Iterable[dict]) -> list[dict]:
    """把流式的 tool_call_chunks（可能被切成多片）拼回完整的工具调用。

    流式模式下模型会把一次工具调用拆成多个 chunk，`name` / `args` / `id`
    落在不同块里，args 甚至是半截 JSON。不回拼就会丢掉整次工具调用 ——
    这正是之前录制流式时静默丢工具调用的原因。
    """
    grouped: dict[int, dict[str, str]] = {}
    for chunk in chunks:
        index = chunk.get("index", 0)
        slot = grouped.setdefault(index, {"name": "", "args": "", "id": ""})
        slot["name"] += chunk.get("name") or ""
        slot["args"] += chunk.get("args") or ""
        slot["id"] += chunk.get("id") or ""

    calls: list[dict] = []
    for index in sorted(grouped):
        slot = grouped[index]
        raw_args = slot["args"]
        try:
            args: Any = json.loads(raw_args) if raw_args else {}
        except ValueError:
            args = raw_args          # 拼接失败就保留原文，至少不丢信息
        calls.append({"name": slot["name"], "args": args, "id": slot["id"]})
    return calls


def _assert_replayable(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass
class Cassette:
    """一个录像文件。

    path: 文件路径
    mode: "record" 或 "replay"
    redact: 可选，写入前对请求/响应做脱敏的回调（例如抹掉手机号）。
        **注意**：脱敏只影响落盘内容，不影响指纹——指纹在脱敏前已算好，
        因此脱敏后依然能正确回放。
    """

    path: Path | str
    mode: str = "replay"
    redact: Callable[[Any], Any] | None = None
    _entries: dict[str, list[dict]] = field(default_factory=dict, repr=False)
    _cursor: dict[str, int] = field(default_factory=dict, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES} 之一，收到 {self.mode!r}")
        if self.mode == "replay" and not self.path.exists():
            raise FileNotFoundError(
                f"回放需要已有的录像文件：{self.path}\n"
                "先用 mode='record' 录一次，或检查路径。"
            )
        if self.path.exists():
            self._load()

    # ---------- 文件 ----------

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        version = data.get("version")
        if version != CASSETTE_VERSION:
            raise ValueError(f"不支持的录像格式版本：{version}（期望 {CASSETTE_VERSION}）")
        recorded = data.get("langchain_core")
        current = _langchain_core_version()
        if recorded and recorded != current:
            # 不直接失败：多数升级是兼容的。但要提醒——消息序列化一变，指纹就会
            # 全变、所有录像失效，而报错只会说"查不到记录"，不提醒根本想不到
            # 是依赖升级导致的。
            logger.warning(
                "录像由 langchain-core %s 录制，当前为 %s。"
                "若回放报 CassetteMissing，先怀疑依赖升级导致指纹变化。",
                recorded, current,
            )
        for entry in data.get("entries", []):
            self._entries.setdefault(entry["key"], []).append(entry)

    def save(self) -> None:
        with self._lock:
            entries = [entry for key in self._entries for entry in self._entries[key]]
            payload = {
                "version": CASSETTE_VERSION,
                "langchain_core": _langchain_core_version(),
                "entries": entries,
            }
            text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 原子替换：先写临时文件再 os.replace。直接覆盖的话，进程中途崩溃
            # 会留下半截 JSON，整个录像文件就废了。
            handle, temp_name = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=".cassette-", suffix=".tmp"
            )
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    stream.write(text + "\n")
                os.replace(temp_name, self.path)
            except BaseException:
                Path(temp_name).unlink(missing_ok=True)
                raise

    # ---------- 记录 / 取用 ----------

    def put(self, key: str, request: dict, response: dict, note: str = "") -> None:
        if self.redact is not None:
            request = self.redact(request)
            response = self.redact(response)
        entry: dict[str, Any] = {"key": key, "request": request, "response": response}
        if note:
            entry["note"] = note
        with self._lock:
            self._entries.setdefault(key, []).append(entry)

    def take(self, key: str) -> dict:
        entries = self._entries.get(key)
        used = self._cursor.get(key, 0)
        if not entries or used >= len(entries):
            raise CassetteMissing(
                f"没有可用的录制记录（键 {key}，已消费 {used} 条）。"
                "输入可能变了（提示词 / 工具 / 历史 / 模型名），"
                "请用 mode='record' 重新录制。"
            )
        self._cursor[key] = used + 1
        return entries[used]

    # ---------- 元信息 ----------

    @property
    def entry_count(self) -> int:
        return sum(len(items) for items in self._entries.values())

    def keys(self) -> list[str]:
        return list(self._entries)

    def reset_cursor(self) -> None:
        self._cursor.clear()

    def consumed(self) -> int:
        return sum(self._cursor.values())


def _response_to_dict(message: Any) -> dict:
    return {
        "content": _assert_replayable(getattr(message, "content", "")),
        "tool_calls": [
            {"name": call.get("name"), "args": call.get("args", {}), "id": call.get("id")}
            for call in (getattr(message, "tool_calls", None) or [])
        ],
    }


def _dict_to_message(payload: dict) -> AIMessage:
    return AIMessage(
        content=payload.get("content", ""),
        tool_calls=payload.get("tool_calls") or [],
    )


class CassetteModel:
    """套在真实模型外面的录像机。

    - record：调用真实模型，把「输入 → 输出」写进录像；
    - replay：不碰网络，按输入指纹查表返回当时的输出。
    """

    def __init__(self, cassette: Cassette, real: Any = None,
                 tool_names: tuple[str, ...] = (),
                 tool_objects: Any = None, bind_kwargs: dict | None = None,
                 model_name: str = "unknown", replay_chunk: int = 8,
                 extra: str = ""):
        self.cassette = cassette
        self._real = real
        self._tool_names = tool_names
        self._tool_objects = tool_objects
        self._bind_kwargs = bind_kwargs or {}
        self.model_name = model_name
        self.replay_chunk = replay_chunk
        self._extra = extra

    # ---------- 与 LangChain 对齐 ----------

    def bind_tools(self, tools: Any, **kwargs: Any) -> "CassetteModel":
        names = tuple(getattr(tool, "name", str(tool)) for tool in tools)
        return self._spawn(tool_names=names, tool_objects=tools, bind_kwargs=kwargs)

    def with_structured_output(self, schema: Any = None, **kwargs: Any) -> "CassetteStructuredModel":
        name = getattr(schema, "__name__", None) or str(schema)
        return CassetteStructuredModel(
            cassette=self.cassette, real=self._real, model_name=self.model_name,
            schema=schema, schema_name=name, structured_kwargs=kwargs,
            tool_names=self._tool_names,
        )

    # ---------- 内部 ----------

    def _spawn(self, **overrides: Any) -> "CassetteModel":
        params = dict(
            cassette=self.cassette, real=self._real, tool_names=self._tool_names,
            tool_objects=self._tool_objects, bind_kwargs=self._bind_kwargs,
            model_name=self.model_name, replay_chunk=self.replay_chunk, extra=self._extra,
        )
        params.update(overrides)
        return CassetteModel(**params)

    def _key(self, messages: Any) -> str:
        return request_key(self.model_name, messages, self._tool_names, self._extra)

    def _bound_real(self) -> Any:
        if self._real is None:
            return None
        if self._tool_objects is not None:
            return self._real.bind_tools(self._tool_objects, **self._bind_kwargs)
        return self._real

    def _request_payload(self, messages: Any) -> dict:
        return {
            "model": self.model_name,
            "tools": list(self._tool_names),
            "messages": [message_to_dict(message) for message in messages],
        }

    # ---------- 调用 ----------

    def invoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        key = self._key(messages)
        if self.cassette.mode == "record":
            response = self._bound_real().invoke(messages, **kwargs)
            self.cassette.put(key, self._request_payload(messages), _response_to_dict(response))
            self.cassette.save()
            return response
        return _dict_to_message(self.cassette.take(key)["response"])

    async def astream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """流式录制与回放。

        录制时**原样透传**真实模型的 chunk（不再抽取字符串），因为流式里除了文本，
        还可能有 `tool_call_chunks`——模型会把一次工具调用拆成多个块，args 甚至是
        半截 JSON。早期版本只取 `chunk.content`，于是**流式下的工具调用被静默丢弃**，
        回放时模型"什么都不想做"，工具永远调不到。

        回放时按记录重建：文本按 `replay_chunk` 切块，工具调用作为最后一个 chunk 发出。
        """
        key = self._key(messages)
        if self.cassette.mode == "record":
            pieces: list[str] = []
            tool_chunks: list[dict] = []
            async for chunk in self._bound_real().astream(messages, **kwargs):
                pieces.append(_content_text(chunk))
                for tool_chunk in (getattr(chunk, "tool_call_chunks", None) or []):
                    tool_chunks.append(dict(tool_chunk))
                yield chunk          # 透传，调用方拿到的是真实对象

            self.cassette.put(
                key, self._request_payload(messages),
                {
                    "content": "".join(pieces),
                    "tool_calls": _assemble_tool_calls(tool_chunks),
                    "tool_call_chunks": tool_chunks,
                    "streamed": True,
                },
            )
            self.cassette.save()
            return

        recorded = self.cassette.take(key)["response"]
        text = recorded.get("content", "")
        for start in range(0, len(text), self.replay_chunk):
            yield AIMessageChunk(content=text[start:start + self.replay_chunk])

        tool_chunks = recorded.get("tool_call_chunks")
        if tool_chunks:
            yield AIMessageChunk(content="", tool_call_chunks=tool_chunks)


class CassetteStructuredModel:
    """结构化输出（with_structured_output）的录制与回放。

    录制时把解析后的对象序列化；回放时用同一个 schema 重建，所以调用方拿到的
    类型与真实运行时一致（pydantic 模型得到模型实例，dict schema 得到 dict）。
    """

    def __init__(self, cassette: Cassette, real: Any, model_name: str,
                 schema: Any, schema_name: str, structured_kwargs: dict,
                 tool_names: tuple[str, ...] = ()):
        self.cassette = cassette
        self._real = real
        self.model_name = model_name
        self.schema = schema
        self._schema_name = schema_name
        self._structured_kwargs = structured_kwargs
        self._tool_names = tool_names

    def _key(self, messages: Any) -> str:
        return request_key(
            self.model_name, messages, self._tool_names, extra=f"structured:{self._schema_name}"
        )

    def invoke(self, messages: Any, **kwargs: Any):
        key = self._key(messages)
        if self.cassette.mode == "record":
            chain = self._real.with_structured_output(self.schema, **self._structured_kwargs)
            result = chain.invoke(messages, **kwargs)
            payload = {
                "kind": "pydantic" if hasattr(result, "model_dump") else "raw",
                "data": result.model_dump() if hasattr(result, "model_dump") else result,
            }
            self.cassette.put(
                key,
                {"model": self.model_name, "tools": list(self._tool_names),
                 "schema": self._schema_name,
                 "messages": [message_to_dict(m) for m in messages]},
                payload,
            )
            self.cassette.save()
            return result

        payload = self.cassette.take(key)["response"]
        if payload.get("kind") == "pydantic":
            return self.schema(**payload["data"])
        return payload.get("data")

    def astream(self, messages: Any, **kwargs: Any):
        # 刻意用普通函数而非 async 生成器：async 生成器的函数体要等被迭代才执行，
        # 那样调用方会拿到一个"看起来正常"的对象、错误被推迟到运行中。
        # 这里在调用时就明确失败。
        raise NotImplementedError(
            "结构化输出不支持流式：结构化结果必须收完整才能解析。"
            "请改用 .invoke()。"
        )


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------

def wrap_model(real: Any = None, cassette: Cassette | str | Path | None = None,
               mode: str | None = None, model_name: str | None = None) -> Any:
    """给模型套上录像机。

    三种用法：

        # 1) 显式传录像对象
        wrap_model(real, Cassette("x.json", "record"))

        # 2) 传路径 + 模式
        wrap_model(real, "x.json", mode="record")

        # 3) 什么都不传：读环境变量 AGENT_CASSETTE / AGENT_CASSETTE_MODE
        wrap_model(real)          # 未设置环境变量时原样返回 real

    第三种是为了"生产代码零改动"——只在测试环境设两个环境变量即可。
    """
    if cassette is None:
        env_path = os.getenv(ENV_PATH)
        if not env_path:
            return real
        cassette = Cassette(env_path, os.getenv(ENV_MODE, "replay"))
    elif not isinstance(cassette, Cassette):
        cassette = Cassette(cassette, mode or "replay")

    resolved_name = model_name or getattr(real, "model_name", None) or "unknown"
    return CassetteModel(cassette, real, model_name=str(resolved_name))


class _Session:
    def __init__(self, path: str | Path, mode: str):
        self.cassette = Cassette(path, mode)

    def wrap(self, real: Any, model_name: str | None = None) -> Any:
        name = model_name or getattr(real, "model_name", None) or "unknown"
        return CassetteModel(self.cassette, real, model_name=str(name))

    def __enter__(self) -> Cassette:
        return self.cassette

    def __exit__(self, *exc_info: Any) -> None:
        if self.cassette.mode == "record":
            self.cassette.save()


def record_session(path: str | Path) -> _Session:
    """with record_session("x.json") as cas: ... 结束后自动落盘。"""
    return _Session(path, "record")


def replay_session(path: str | Path) -> _Session:
    """with replay_session("x.json") as cas: ... 只读回放。"""
    return _Session(path, "replay")
