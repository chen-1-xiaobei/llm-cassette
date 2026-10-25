"""命令行入口。

    llm-cassette inspect  x.json                # 看录像里有什么
    llm-cassette diff     a.json b.json         # 对比两份录像的模型行为
    llm-cassette record   x.json -- python app.py   # 用录像模式跑任意命令
    llm-cassette replay   x.json -- python app.py   # 用回放模式跑（不联网、不花钱）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .cassette import ENV_MODE, ENV_PATH, Cassette
from .diff import diff_files, render, render_entries


def _cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"文件不存在：{path}", file=sys.stderr)
        return 2
    print(render_entries(path))
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    for candidate in (args.old, args.new):
        if not Path(candidate).exists():
            print(f"文件不存在：{candidate}", file=sys.stderr)
            return 2
    diff = diff_files(args.old, args.new)
    print(render(diff))
    # 有差异返回 1，便于接进 CI：`llm-cassette diff a.json b.json || echo 行为变了`
    return 0 if not diff.has_change else 1


def _cmd_run(mode: str, args: argparse.Namespace) -> int:
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("需要在 -- 之后给出要执行的命令，例如：", file=sys.stderr)
        print(f"  llm-cassette {mode} x.json -- python my_agent.py", file=sys.stderr)
        return 2

    path = Path(args.file)
    if mode == "replay" and not path.exists():
        print(f"回放需要已有录像：{path}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env[ENV_PATH] = str(path.resolve())
    env[ENV_MODE] = mode
    print(f"[llm-cassette] mode={mode} cassette={path}", file=sys.stderr)
    print(f"[llm-cassette] 提示：在被测代码里用 wrap_model(model) 即可生效", file=sys.stderr)
    return subprocess.call(command, env=env)


def _cmd_count(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"文件不存在：{path}", file=sys.stderr)
        return 2
    cassette = Cassette(path, "replay")
    print(f"{path}: {cassette.entry_count} 条记录，{len(cassette.keys())} 个不同指纹")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-cassette",
        description="录制并回放 LLM 交互，让 Agent 行为可确定性测试",
    )
    sub = parser.add_subparsers(dest="command_name", required=True)

    inspect = sub.add_parser("inspect", help="列出录像里的每一条调用")
    inspect.add_argument("file")
    inspect.set_defaults(func=_cmd_inspect)

    count = sub.add_parser("count", help="只统计条数")
    count.add_argument("file")
    count.set_defaults(func=_cmd_count)

    diff = sub.add_parser("diff", help="对比两份录像的模型行为（有差异时退出码为 1）")
    diff.add_argument("old")
    diff.add_argument("new")
    diff.set_defaults(func=_cmd_diff)

    for mode, help_text in (
        ("record", "用录像模式跑命令：真实调用模型并写进文件"),
        ("replay", "用回放模式跑命令：不联网、不花钱"),
    ):
        runner = sub.add_parser(mode, help=help_text)
        runner.add_argument("file")
        runner.add_argument("command", nargs=argparse.REMAINDER, metavar="-- COMMAND")
        runner.set_defaults(func=lambda ns, m=mode: _cmd_run(m, ns))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
