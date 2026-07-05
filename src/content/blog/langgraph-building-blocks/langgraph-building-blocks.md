---
title: "LangGraph Building Blocks：从 State 到运行图"
description: "整理 LangChain 与 LangGraph 的关系，以及 State、Nodes、Edges、StateGraph 的建图与运行要点"
publishDate: 2026-07-05
category: technique
tags: ["langgraph", "langchain", "agent", "workflow"]
draft: false
comment: true
---

## LangChain 和 LangGraph 是什么

先把两个名字分清楚。`LangChain` 是一个用于构建 LLM 应用的框架，重点在于把模型、提示词、工具、检索、结构化输出、agent 等能力组织起来。它适合快速搭建“语言模型应用”的上层逻辑，例如：接收用户输入、调用模型、必要时调用工具、返回结构化结果。

`LangGraph` 则更偏向底层编排。它把一次 agent 或 workflow 的执行过程表示成“有状态的图”：节点负责做计算，边负责决定下一步，状态负责承载中间结果。相比只写一串线性函数调用，LangGraph 更适合表达分支、循环、人工介入、流式观察、持久化状态等更复杂的流程。

二者的关系可以这样理解：

- `LangChain` 面向 LLM 应用开发，抽象层次更高。
- `LangGraph` 面向 stateful agent / workflow 的运行时和控制流，抽象层次更底。
- 官方 LangChain agent 现在构建在 LangGraph 之上，因此 LangGraph 可以看作复杂 agent 的执行底座之一。
- 使用 LangGraph 时不一定要使用完整的 LangChain；一个普通 Python 函数也可以成为 LangGraph 节点。

本文关注的是 tutorial1 里最基础、也最需要记住的部分：`State`、`Nodes`、`Edges` 和 `StateGraph`。

## 基本元素

### State

`State` 是图中流动的数据。每个节点运行时都会收到当前 state，并可以返回一部分 state 更新。

在 tutorial1 里，最小 state 只有一个字符串字段：

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

这不是说 LangGraph 的 state 只能是字符串，而是先用最小例子说明机制。真实项目里，state 通常会包含输入、分类结果、工具调用结果、最终回复、调试信息等字段。例如 ticket 分类任务可以有 `ticket`、`category`、`reply` 三个字段。

需要注意：`TypedDict` 只是定义 state schema 的一种常见方式。LangGraph 也支持其他 schema 形式。tutorial1 使用 `TypedDict`，是因为它轻量、直观，足够说明“有哪些字段会在图里流动”。

### Nodes

`Nodes` 是图里的计算单元。最常见的节点就是一个 Python 函数：接收 `state`，做一些事情，然后返回一个 `dict`。

关键点是：节点返回的 `dict` 不是整个 graph，也不是“下一步去哪”。它表示“对 state 的更新”。例如：

```python
def node_1(state: State) -> dict:
    return {"graph_state": state["graph_state"] + " I am"}
```

为什么可以直接把输出放在 `{}` 里？因为 LangGraph 约定节点返回值是 state update。`{"graph_state": ...}` 的意思是更新 `graph_state` 这个字段。对于没有自定义 reducer 的普通字段，新的字段值会覆盖旧值；如果字段配置了 reducer，则会按 reducer 的规则合并。

所以节点函数的记忆模板是：

1. 从 `state` 里读取需要的信息。
2. 完成一次明确的工作，例如清洗文本、调用模型、分类、生成回复。
3. 返回只包含被更新字段的 `dict`。

### Edges

`Edges` 决定节点之间如何流动。

直接边表示固定路径：从 A 运行到 B。比如 `builder.add_edge("clean_ticket", "classify_ticket")` 表示清洗完成后总是进入分类节点。

条件边表示分支路径：先运行一个 router 函数，再根据 router 的返回值决定下一步去哪个节点。tutorial1 中的 `decide_mood` 用随机数在 `node_2` 和 `node_3` 之间分流，这是为了展示条件边的最小写法；真实项目里，router 通常应该读取 state 中的字段来做确定性判断。

### StateGraph

`StateGraph` 是图的构建器。它把 state schema、节点函数和边连接在一起。常见流程是：

1. `builder = StateGraph(State)`：创建一个以 `State` 为 schema 的图构建器。
2. `add_node`：注册节点名和节点函数。
3. `add_edge`：添加固定路径。
4. `add_conditional_edges`：添加条件路径。
5. `compile()`：编译成可运行的 graph。

可以把 `StateGraph` 理解成“还没运行的蓝图”。只有编译以后，才得到可以 `.invoke()` 或 `.stream()` 的 runnable graph。

## 建图的思路

### 1. 如何定义 state

定义 state 时，不要先想“我有几个函数”，而要先想“图执行过程中有哪些信息必须被记住”。

通常至少分三类：

- 输入字段：用户原始输入、文件内容、ticket 文本等。
- 中间字段：分类结果、路由原因、检索结果、工具返回值等。
- 输出字段：最终回复、结构化结果、报告正文等。

如果某个 router 要根据分类结果决定下一步，那么 `category` 就应该出现在 state 里，并且应由前面的分类节点写入。如果你还想知道为什么走某条边，可以额外加 `route_reason`。tutorial1 的扩展练习就是这个思路：不要让 router “顺手更新 state”，而是在路由前加一个分类节点，把原因写入 state，再让 router 只负责选择下一节点。

### 2. 如何写节点，为什么输出可以直接放在 `{}`

节点应该尽量只做一件事。下面是一个 ticket 清洗和分类的例子：

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph

class TicketState(TypedDict):
    ticket: str
    category: str
    reply: str

def clean_ticket(state: TicketState) -> dict:
    return {"ticket": state["ticket"].strip()}

def classify_ticket(state: TicketState) -> dict:
    text = state["ticket"].lower()
    if any(word in text for word in ["bug", "error", "failed", "broken", "crash", "crashes"]):
        return {"category": "bug"}
    if any(word in text for word in ["feature", "add", "support", "integrate", "integration"]):
        return {"category": "feature"}
    return {"category": "question"}
```

这里的 `return {"category": "bug"}` 是合法的，因为节点函数的返回值会被 LangGraph 当作 state update。它只更新 `category`，不会自动删除 `ticket` 和 `reply`。这也是为什么节点不需要返回完整 state：只返回变化的字段即可。

在工程实践中，最好不要依赖节点内部原地修改 `state`。更清晰的写法是读取 state，然后显式返回一个更新字典。

### 3. 怎么写 router，为什么输出可以是 `str`

router 是条件边使用的函数。它读取当前 state，返回下一步的路径。

```python
def route_ticket(state: TicketState) -> Literal["bug_reply", "feature_reply", "question_reply"]:
    if state["category"] == "bug":
        return "bug_reply"
    if state["category"] == "feature":
        return "feature_reply"
    return "question_reply"
```

为什么 router 可以只返回字符串？因为在 `add_conditional_edges("classify_ticket", route_ticket)` 这种写法下，LangGraph 会把 router 的返回值解释为目标节点名。也就是说，返回 `"bug_reply"` 就表示下一步进入名为 `bug_reply` 的节点。

`Literal[...]` 不是 LangGraph 运行所必需的魔法，而是 Python 类型标注。它能提醒自己和编辑器：这个 router 只应该返回这些节点名。如果返回一个没有注册过的节点名，图在编译或运行时会暴露问题。

如果不希望 router 直接返回节点名，也可以使用 path map，把业务标签映射到节点名。例如 router 返回 `"bug"`，再通过映射让 `"bug"` 指向 `"bug_reply"`。tutorial1 采用的是最直接的版本：返回值本身就是节点名。

### 4. 怎么建图：`add_node`、`add_edge`、`add_conditional_edges`

建图时要把节点和边分开看。先注册所有可能被调用的节点，再描述它们之间的连接关系：

```python
def bug_reply(state: TicketState) -> dict:
    return {"reply": "Thanks for the bug report. We will investigate it."}

def feature_reply(state: TicketState) -> dict:
    return {"reply": "Thanks for the feature request. We will review it."}

def question_reply(state: TicketState) -> dict:
    return {"reply": "Thanks for the question. We will follow up with more details."}

builder = StateGraph(TicketState)

builder.add_node("clean_ticket", clean_ticket)
builder.add_node("classify_ticket", classify_ticket)
builder.add_node("bug_reply", bug_reply)
builder.add_node("feature_reply", feature_reply)
builder.add_node("question_reply", question_reply)

builder.add_edge(START, "clean_ticket")
builder.add_edge("clean_ticket", "classify_ticket")
builder.add_conditional_edges("classify_ticket", route_ticket)
builder.add_edge("bug_reply", END)
builder.add_edge("feature_reply", END)
builder.add_edge("question_reply", END)

ticket_graph = builder.compile()
```

这段图的结构是：

`START -> clean_ticket -> classify_ticket -> 条件分支 -> reply 节点 -> END`

几个 API 的职责要记清楚：

- `add_node(name, fn)`：给函数注册一个图里的节点名。之后边引用的是节点名，不是直接引用函数对象。
- `add_edge(from_node, to_node)`：添加固定边。运行到 `from_node` 后，总是进入 `to_node`。
- `add_conditional_edges(from_node, router)`：添加条件边。运行到 `from_node` 后，调用 router，根据它的返回值选择下一节点。
- `START`：图的入口标记，不是业务节点。
- `END`：图的结束标记，也不是业务节点。

### 5. 运行图：`.compile()`、`.invoke()`、`.stream()`

`.compile()` 会把 `StateGraph` 构建器编译成可运行对象。编译阶段会进行结构检查，并把图转换成 LangGraph 的运行时对象。一般来说，所有节点和边都添加好以后，再调用一次 `compile()`。

`.invoke(input_state)` 表示运行图直到结束，并返回最终 state：

```python
result = ticket_graph.invoke({
    "ticket": "  The app crashes when I upload a PDF.  ",
    "category": "",
    "reply": "",
})
```

如果图的路径是 bug 分支，最终结果会保留清洗后的 `ticket`、分类得到的 `category`，以及回复节点写入的 `reply`。

`.stream(input_state)` 用于观察图执行过程中的中间输出。tutorial1 里用它打印每一步节点输出，适合调试复杂 agent：

```python
for step in ticket_graph.stream({
    "ticket": "Can you add Slack integration?",
    "category": "",
    "reply": "",
}):
    print(step)
```

要注意，`.stream()` 的 chunk 形状会受 LangGraph 版本和 `stream_mode` 参数影响。学习 building blocks 时，只要先记住它的用途：`invoke` 看最终结果，`stream` 看执行过程。

## 最后复盘

LangGraph 的最小心智模型可以压缩成一句话：用 `State` 描述图里流动的数据，用 `Node` 更新数据，用 `Edge` 决定控制流，用 `StateGraph` 把它们组装并编译成可运行图。

再进一步记住两个返回值约定：

1. 节点返回 `dict`，因为它返回的是 state update。
2. router 返回 `str`，因为在最直接的条件边写法里，这个字符串就是下一节点名。

只要这两个约定清楚，`add_node`、`add_edge`、`add_conditional_edges`、`compile`、`invoke`、`stream` 的关系就会自然很多。

## 参考资料

- [LangChain Python overview](https://docs.langchain.com/oss/python/langchain/overview)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
