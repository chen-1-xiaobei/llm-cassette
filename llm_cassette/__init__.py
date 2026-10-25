"""llm-cassette：录制并回放 LLM 交互，让 Agent 的行为可确定性测试。

快速开始
--------

录制（真实调用模型，一次）：:

    from langchain_deepseek import ChatDeepSeek
    from llm_cassette import record_session

    real = ChatDeepSeek(model="deepseek-chat", temperature=0)
    with record_session("session.json"):
        model = ...  # 见下方 wrap

回放（不联网、不花钱、逐字节复现）：:

    with replay_session("session.json") as cassette:
        model = cassette.wrap(ChatDeepSeek(model="deepseek-chat"))

生产代码零改动：在环境里设两个变量即可，其余交给 `wrap_model`：

    LLM_CASSETTE=session.json
    LLM_CASSETTE_MODE=replay

    model = wrap_model(ChatDeepSeek(model="deepseek-chat"))

命令行
------

    llm-cassette inspect x.json          # 看录像里有什么
    llm-cassette diff a.json b.json       # 对比两次运行的行为差异
    llm-cassette record x.json -- python app.py
    llm-cassette replay x.json -- python app.py
"""
from .cassette import (
    ENV_MODE,
    ENV_PATH,
    CASSETTE_VERSION,
    Cassette,
    CassetteMissing,
    CassetteModel,
    CassetteStructuredModel,
    record_session,
    replay_session,
    wrap_model,
)
from .diff import CassetteDiff, diff_files, describe, render
from .keys import message_to_dict, request_key

__version__ = "0.1.0"

__all__ = [
    "CASSETTE_VERSION",
    "ENV_MODE",
    "ENV_PATH",
    "Cassette",
    "CassetteDiff",
    "CassetteMissing",
    "CassetteModel",
    "CassetteStructuredModel",
    "describe",
    "diff_files",
    "message_to_dict",
    "record_session",
    "render",
    "replay_session",
    "request_key",
    "wrap_model",
    "__version__",
]
