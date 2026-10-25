"""对比两份录像：换模型、改提示词之后，模型行为到底变了什么。

这是整套东西最实用的一处——"回归测试"只告诉你"变了没"，`diff` 告诉你"**变了什么**"。

例如把 deepseek-chat 换成另一个模型后：

    $ llm-cassette diff before.json after.json
    ~ 8ad8b3b526c933d8  第 1 轮检索
        改前: 调用工具: search_knowledge_base({"question": "出差住宿报销标准是多少"})
        改后: 调用工具: search_knowledge_base({"question": "住宿标准"})
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cassette import CASSETTE_VERSION


CHANGED = "changed"
ONLY_OLD = "only_old"
ONLY_NEW = "only_new"


@dataclass
class EntryDiff:
    key: str
    kind: str
    old: dict | None
    new: dict | None
    old_descs: list[str] | None = None     # 同一指纹多次调用时的逐条描述
    new_descs: list[str] | None = None


@dataclass
class CassetteDiff:
    same: list[str] = field(default_factory=list)
    changed: list[EntryDiff] = field(default_factory=list)
    only_old: list[EntryDiff] = field(default_factory=list)
    only_new: list[EntryDiff] = field(default_factory=list)

    @property
    def has_change(self) -> bool:
        return bool(self.changed or self.only_old or self.only_new)

    def summary(self) -> str:
        return (f"未变 {len(self.same)} 条，"
                f"输出变化 {len(self.changed)} 条，"
                f"仅旧录像有 {len(self.only_old)} 条，"
                f"仅新录像有 {len(self.only_new)} 条")


def describe(response: dict) -> str:
    """把一条响应压成一行可读的描述。"""
    calls = response.get("tool_calls") or []
    if calls:
        rendered = [
            f'{call.get("name")}({json.dumps(call.get("args", {}), ensure_ascii=False, sort_keys=True)})'
            for call in calls
        ]
        return "调用工具: " + "; ".join(rendered)
    content = (response.get("content") or "").replace("\n", " ").strip()
    if response.get("streamed"):
        return f"流式文本({len(content)} 字): {content[:60]}"
    return "文本回复: " + (content[:60] if content else "(空)")


def _load(path: Path) -> dict[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version")
    if version != CASSETTE_VERSION:
        raise ValueError(f"{path}: 不支持的录像格式版本 {version}（期望 {CASSETTE_VERSION}）")
    entries: dict[str, list[dict]] = {}
    for entry in data.get("entries", []):
        # 同一指纹可能对应多次调用，全部保留 —— 只取第一条会漏掉后面的变化
        entries.setdefault(entry["key"], []).append(entry)
    return entries


def diff_files(old_path: str | Path, new_path: str | Path) -> CassetteDiff:
    old = _load(Path(old_path))
    new = _load(Path(new_path))
    result = CassetteDiff()

    for key, old_entries in old.items():
        new_entries = new.get(key)
        if new_entries is None:
            result.only_old.append(EntryDiff(key, ONLY_OLD, old_entries[0], None))
            continue
        old_descs = [describe(entry.get("response", {})) for entry in old_entries]
        new_descs = [describe(entry.get("response", {})) for entry in new_entries]
        if old_descs == new_descs:
            result.same.append(key)
        else:
            result.changed.append(EntryDiff(
                key, CHANGED, old_entries[0], new_entries[0], old_descs, new_descs,
            ))

    for key, new_entries in new.items():
        if key not in old:
            result.only_new.append(EntryDiff(key, ONLY_NEW, None, new_entries[0]))

    return result


def render(diff: CassetteDiff) -> str:
    lines = [diff.summary()]
    for item in diff.changed:
        lines.append("")
        lines.append(f"~ {item.key}  （模型输出变了）")
        old_descs = item.old_descs or [describe(item.old["response"])]
        new_descs = item.new_descs or [describe(item.new["response"])]
        many = len(old_descs) > 1 or len(new_descs) > 1
        indent = "        " if many else "    "
        for index in range(max(len(old_descs), len(new_descs))):
            before = old_descs[index] if index < len(old_descs) else "(无)"
            after = new_descs[index] if index < len(new_descs) else "(无)"
            if many:
                lines.append(f"    第 {index + 1} 次调用")
            lines.append(f"{indent}改前: {before}")
            lines.append(f"{indent}改后: {after}")
    for item in diff.only_old:
        lines.append("")
        lines.append(f"- {item.key}  （只在旧录像里：输入变了，或这轮调用不再发生）")
        lines.append(f"    旧: {describe(item.old['response'])}")
    for item in diff.only_new:
        lines.append("")
        lines.append(f"+ {item.key}  （只在新录像里：新增了一轮调用）")
        lines.append(f"    新: {describe(item.new['response'])}")
    if not diff.has_change:
        lines.append("两份录像的模型行为一致。")
    return "\n".join(lines)


def render_entries(path: str | Path) -> str:
    """列出录像里的每一条（`inspect` 用）。"""
    entries = _load(Path(path))
    total = sum(len(items) for items in entries.values())
    lines = [f"{path}: 共 {total} 条（{len(entries)} 个不同指纹）"]
    index = 0
    for key, items in entries.items():
        for entry in items:
            index += 1
            request = entry.get("request", {})
            messages = request.get("messages", [])
            last_user = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "human"), ""
            )
            tools = request.get("tools") or []
            lines.append("")
            lines.append(f"[{index}] {key}")
            lines.append(f"    工具: {tools or '(无)'}   消息数: {len(messages)}")
            if last_user:
                lines.append(f"    最后一句用户: {last_user[:50]}")
            lines.append(f"    → {describe(entry.get('response', {}))}")
    return "\n".join(lines)
