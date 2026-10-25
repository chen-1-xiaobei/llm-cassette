"""命令行：inspect / count / diff。"""
from langchain_core.messages import AIMessage, HumanMessage

from _fakes import FakeChatModel, tool_call
from llm_cassette import Cassette, CassetteModel
from llm_cassette.cli import main


def _record(path, responses, question="q", model_name="m"):
    model = CassetteModel(Cassette(path, "record"), FakeChatModel(responses), model_name=model_name)
    model.invoke([HumanMessage(question)])


def test_inspect_lists_entries(tmp_path, capsys):
    path = tmp_path / "c.json"
    _record(path, [tool_call("search", {"q": "住宿标准"})])

    assert main(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "共 1 条" in out
    assert "search" in out and "住宿标准" in out


def test_count(tmp_path, capsys):
    path = tmp_path / "c.json"
    _record(path, [AIMessage("a")])
    _record(path, [AIMessage("b")], question="第二个问题")

    assert main(["count", str(path)]) == 0
    out = capsys.readouterr().out
    assert "2 条记录" in out


def test_diff_exit_code_signals_change(tmp_path, capsys):
    same_a, same_b = tmp_path / "a.json", tmp_path / "b.json"
    _record(same_a, [AIMessage("一致")])
    _record(same_b, [AIMessage("一致")])
    assert main(["diff", str(same_a), str(same_b)]) == 0

    changed = tmp_path / "c.json"
    _record(changed, [AIMessage("不一致")])
    assert main(["diff", str(same_a), str(changed)]) == 1, "有差异时退出码应为 1，便于接 CI"


def test_missing_file_returns_2(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "nope.json")]) == 2
    assert main(["count", str(tmp_path / "nope.json")]) == 2
    assert main(["diff", str(tmp_path / "nope.json"), str(tmp_path / "nope2.json")]) == 2


def test_run_requires_command(tmp_path, capsys):
    path = tmp_path / "c.json"
    _record(path, [AIMessage("x")])
    assert main(["replay", str(path)]) == 2
    assert "需要在 -- 之后给出要执行的命令" in capsys.readouterr().err


def test_record_passes_env_to_subprocess(tmp_path, capfd):
    """record 会把两个环境变量传进子进程，被测代码据此自动套上录像机。

    用 capfd 而不是 capsys：子进程的 stdout 属于文件描述符层，
    capsys 只替换 sys.stdout，抓不到子进程输出。
    """
    import sys

    path = tmp_path / "c.json"
    script = "import os; print(os.environ.get('LLM_CASSETTE_MODE'), os.environ.get('LLM_CASSETTE'))"

    assert main(["record", str(path), "--", sys.executable, "-c", script]) == 0
    out, _ = capfd.readouterr()
    assert "record" in out and str(path) in out


def test_replay_refuses_missing_cassette(tmp_path, capfd):
    import sys

    assert main(["replay", str(tmp_path / "missing.json"), "--",
                 sys.executable, "-c", "pass"]) == 2
    _, err = capfd.readouterr()
    assert "回放需要已有录像" in err


def test_replay_passes_env_when_cassette_exists(tmp_path, capfd):
    import sys

    path = tmp_path / "c.json"
    Cassette(path, "record").save()          # 先建一个空录像

    script = "import os; print(os.environ.get('LLM_CASSETTE_MODE'))"
    assert main(["replay", str(path), "--", sys.executable, "-c", script]) == 0
    out, _ = capfd.readouterr()
    assert "replay" in out
