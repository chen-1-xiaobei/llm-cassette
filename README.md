# agent-cassette

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CI](https://github.com/chen-1-xiaobei/agent-cassette/actions/workflows/ci.yml/badge.svg)](https://github.com/chen-1-xiaobei/agent-cassette/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**录制并回放 LLM 交互，让 Agent 的行为可以确定性测试——不联网、不花钱、不需要密钥。**

大模型的输出每次都不一样。这带来一个尴尬的局面：你能测工具、能测数据库、能测路由，
**唯独测不了"模型这一步到底做了什么"**。于是 CI 里永远缺一块，改提示词也只能靠手感。

`agent-cassette` 给模型套一层"录像机"：真实跑一次，把「输入 → 输出」存成 JSON；
之后测试不再问模型，而是按输入查表返回当时那一条。

```console
$ agent-cassette diff before.json after.json
未变 0 条，输出变化 1 条，仅旧录像有 0 条，仅新录像有 0 条

~ be2d13005f351a4b  （模型输出变了）
    改前: 调用工具: search_knowledge_base({"question": "出差住宿报销标准是多少", "top_k": 4})
    改后: 调用工具: search_knowledge_base({"question": "住宿标准", "top_k": 4})
```

---

## 安装

```bash
pip install -e ".[dev]"        # 开发
pip install agent-cassette        # 发布后
```

依赖只有 `langchain-core`，支持任何 LangChain 兼容的聊天模型。

## 三种用法

### 1. 显式包一层

```python
from langchain_deepseek import ChatDeepSeek
from agent_cassette import Cassette, wrap_model

real = ChatDeepSeek(model="deepseek-chat", temperature=0)

# 录制
model = wrap_model(real, Cassette("session.json", "record"))
response = model.bind_tools(tools).invoke(messages)

# 回放（不联网、不花钱）
model = wrap_model(ChatDeepSeek(model="deepseek-chat"), Cassette("session.json", "replay"))
response = model.bind_tools(tools).invoke(messages)   # 与录制时逐字节一致
```

### 2. 会话上下文（自动落盘）

```python
from agent_cassette import record_session, replay_session

with record_session("session.json") as cassette:
    model = cassette.wrap(real)
    ...

with replay_session("session.json") as cassette:
    model = cassette.wrap(real)      # 不会真的调用，只查表
```

### 3. 生产代码零改动（推荐）

代码里只写 `wrap_model(model)`，由环境变量决定行为：

```python
from agent_cassette import wrap_model

model = wrap_model(ChatDeepSeek(model="deepseek-chat", temperature=0))
```

```bash
# 录制一次（真实调用）
AGENT_CASSETTE=session.json AGENT_CASSETTE_MODE=record python app.py

# 之后所有测试/CI 都用回放
AGENT_CASSETTE=session.json AGENT_CASSETTE_MODE=replay python app.py
```

未设置 `AGENT_CASSETTE` 时 `wrap_model` **原样返回**，对生产没有任何影响。

## 命令行

```console
$ agent-cassette inspect examples/cassettes/before.json
examples\cassettes\before.json: 共 1 条

[1] be2d13005f351a4b
    工具: ['search_knowledge_base']   消息数: 1
    最后一句用户: 出差住宿一晚最多能报多少钱？
    → 调用工具: search_knowledge_base({"question": "出差住宿报销标准是多少", "top_k": 4})

$ agent-cassette diff a.json b.json      # 有差异时退出码为 1，可直接接 CI
$ agent-cassette record session.json -- python app.py
$ agent-cassette replay session.json -- python app.py
```

`diff` 是这套东西最实用的一处：回归测试只告诉你"变了没"，它告诉你"**变了什么**"。
换模型、升模型、改提示词之后，跑一遍再看 diff，就能知道模型的行为往哪个方向偏了。

## 设计要点

| 决定 | 为什么 |
| --- | --- |
| **查不到记录就抛 `CassetteMissing`** | 绝不静默返回空。"测试全绿但其实什么都没测"比测试失败危险得多 |
| **工具集合参与指纹** | 增删工具即契约变更，旧录像自动失效——这是保护，不是麻烦 |
| **同一输入可存多条，按顺序消费** | 模型对相同输入给出不同输出是正常的；用完再取就报错 |
| **易变元数据不进指纹** | token 用量、响应 ID 每次都不同，算进键会导致回放必然失效 |
| **指纹在脱敏之前算** | 录盘内容可以抹掉手机号等敏感信息，且不影响回放匹配 |
| **不支持的能力明确报错** | 例如结构化输出的流式回放，直接 `NotImplementedError`，不返回"看起来能用"的对象 |

## 覆盖范围

支持：

- `bind_tools(...).invoke()` / `.astream()` —— Agent 决策路径（选工具、写参数、改写查询）
- `with_structured_output(Schema).invoke()` —— 回放返回同样的 pydantic 实例
- 同一进程内任意模型标识、任意工具集合
- 写入前脱敏（`Cassette(..., redact=callable)`）

不支持：

- 结构化输出的**流式**回放（结构化结果必须收完整才能解析）
- 原生流式工具调用（`tool_call_chunks`）的细粒度回放
- 跨进程共享同一个录像对象（每次进程启动独立读文件，游标不共享）

## 它不是什么

| 不是 | 区别 |
| --- | --- |
| **缓存** | 缓存为了快，未命中就重算；这里为了可复现，未命中就报错 |
| **Mock** | Mock 是人手写的"我以为模型会这样答"；这里的记录来自真实模型 |
| **状态管理** | 状态管理参与运行时；这里只在测试里用，生产路径零影响 |
| **对话记忆** | 记忆是给模型看的历史；录像不给模型看，只给测试用 |

## 一个真实的使用场景

这个库是在一个企业知识库 Agent 项目里长出来的，起因是一个具体问题：
**多轮对话里"那住宿标准呢"会被改写成什么，没法自动测。**

现在那条断言变成了：

```python
def test_second_turn_rewrites_reference(replayer):
    _, second = run_scenario(replayer)
    assert second.queries[0] != "那住宿标准呢"
    assert "住宿" in second.queries[0]
```

零密钥、零网络、不花钱，可以直接进 CI。

## 开发

```bash
pip install -e ".[dev]"
pytest
python examples/make_demo.py     # 重新生成演示录像
```

## 许可

[MIT](LICENSE)
