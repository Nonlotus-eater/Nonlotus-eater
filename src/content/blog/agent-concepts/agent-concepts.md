---
title: "Agent Concepts"
description: "整理 Agent、LLM、Tools、MCP 与 Thought-Action-Observation 工作流的基础概念"
publishDate: 2026-07-05
category: technique
tags: ["agent", "llm", "tools", "mcp"]
draft: false
comment: true
---

## Agent 的基本定义

Agent 可以理解为一个借助 AI 模型与外部环境交互的系统。它的目标不是单纯生成一段文本，而是在用户给定的目标下，通过推理、规划和执行动作来完成任务。

这个定义里有三个关键词：

- `Reasoning`：判断当前问题是什么、缺什么信息、下一步应该做什么。
- `Planning`：把一个目标拆成若干步骤，并决定步骤之间的顺序。
- `Acting`：调用工具、查询数据、执行代码、操作环境，真正改变外部状态或获得新信息。

因此，Agent 和普通聊天机器人的区别不在于“是否会说话”，而在于它是否能围绕目标持续决策，并通过外部动作推进任务。

## Agent 的两部分架构

一个最小的 Agent 可以拆成两部分：`Brain` 和 `Body`。

`Brain` 指 AI model，通常是 LLM 或其他具备推理、生成、规划能力的模型。它负责理解用户目标、阅读上下文、选择下一步动作，以及根据观察结果修正计划。

`Body` 指 capabilities and tools，也就是 Agent 能实际使用的能力。例如搜索网页、调用 API、检索数据库、生成图片、运行代码、读写文件等。没有工具时，模型主要只能生成文本；有工具后，Agent 才能通过行动与环境交互。

所以可以把 Agent 记成：

`Agent = AI Model + Tools + Control Loop`

其中 control loop 指持续的“思考、行动、观察”循环。

## LLM：Agent 的常见大脑

LLM，即 Large Language Model，是许多 Agent 的核心模型。LLM 的基本训练目标可以概括为：给定前面的 token 序列，预测下一个 token。

这个目标看起来简单，但规模足够大时，模型会学到语言模式、知识关联、指令跟随、代码结构以及一定程度的推理能力。Agent 正是利用这些能力来理解目标、生成计划和选择工具。

### Tokens

Token 是模型训练和推理时处理的数据单位。一个 token 可能是一个短词、一个长词的一部分、一个标点符号，也可能是 emoji 等特殊符号。

`Tokenization` 是把原始数据转成 token 的过程。文本最容易理解：句子会被切成 token 序列。更广义地说，图像、音频、视频等数据也可以经过编码或离散化，转成模型能够处理的 token 或 token-like 表示。

### Pretraining 与 Post-training

`Pretraining` 的目标是让模型学习预测下一个 token。通过大规模语料训练，模型获得通用语言能力和世界知识。

`Post-training` 是预训练之后的进一步调整，让模型更适合某类任务、组织需求或交互规范。常见技术包括 fine-tuning、pruning、quantization、distillation、reinforcement learning from human feedback，以及 synthetic data augmentation。

这些技术的目的不同：fine-tuning 更偏任务适配，quantization 更偏部署效率，distillation 更偏把大模型能力迁移到小模型，RLHF 更偏对齐人类偏好。

## Tools：Agent 与环境交互的接口

Tool 可以理解为提供给 LLM 调用的函数。它把模型的“想做某件事”变成可执行的外部动作。

常见工具包括：

- Web search：搜索互联网信息。
- Image generation：生成图像。
- Retrieval：从文档库或知识库检索信息。
- API call：调用外部服务。
- Code execution：运行代码或计算结果。

工具不是随便丢给模型就能用。通常需要在 system prompt 或工具 schema 中清楚描述工具的名称、用途、输入参数、参数类型和输出类型。模型根据这些描述决定是否调用工具，以及如何组织参数。

一个工具描述可以抽象成：

```text
Tool Name: calculator
Description: Multiply two integers.
Arguments: a: int, b: int
Output: int
```

这个结构越清晰，模型越容易生成可解析、可执行的调用。

## MCP：标准化工具接入

MCP，即 Model Context Protocol，是一种开放协议，用来标准化应用如何向 LLM 提供工具和上下文。

它解决的问题是：如果每个框架、每个模型供应商、每个工具都设计一套不同的接口，Agent 系统会很难复用。MCP 的意义在于提供统一协议，使支持 MCP 的应用能够连接 MCP server 暴露出来的工具。

MCP 的价值主要体现在三点：

- 预构建集成更多：Agent 理解 MCP 后，可以接入许多已经实现好的 MCP server。
- 更容易切换模型或供应商：工具协议稳定时，模型层可以相对独立替换。
- 更便于在自己的基础设施里控制数据和权限：工具运行在哪里、暴露什么能力、能访问什么数据，都可以被工程化管理。

因此，MCP 可以理解为 Agent 工具生态的标准化接口层。

## Agent Workflow：Thought -> Action -> Observation

Agent 的典型工作流可以写成：

`Thought -> Action -> Observation`

这个循环会重复执行，直到任务完成或达到停止条件。

### Thought

Thought 表示 Agent 的内部推理和规划过程。它回答的问题是：现在知道什么、还缺什么、下一步应该做什么。

常见 Thought 类型包括：

| 类型 | 含义 |
| --- | --- |
| Planning | 把任务拆成步骤，例如先收集数据，再分析趋势，最后生成报告。 |
| Analysis | 根据已有信息判断问题来源，例如从报错推断数据库连接参数有问题。 |
| Decision Making | 在多个选项中做选择，例如根据预算推荐中档方案。 |
| Problem Solving | 设计解决路径，例如先 profile 代码，再优化瓶颈。 |
| Memory Integration | 利用历史上下文，例如记住用户偏好 Python 示例。 |
| Self-Reflection | 评估上一步是否有效，并在失败时换策略。 |
| Goal Setting | 明确完成任务前需要满足的标准。 |
| Prioritization | 判断优先级，例如先修安全漏洞，再加新功能。 |

需要注意，Thought 不等于必须把模型的全部内部推理原样展示给用户。工程上常见做法是保留可解释的计划、简短理由或状态摘要，而不是暴露冗长的隐藏推理过程。

### Chain of Thought

Chain of Thought 是一种提示方法，核心思想是引导模型在给出最终答案前，按步骤思考问题。它对数学推理、复杂问答、规划类任务有帮助。

在产品或工程系统中，更安全的理解方式是：我们希望模型有更好的分步解决能力，但不一定要求它把完整推理链逐字输出。很多时候，输出简洁的结论和必要解释更合适。

### ReAct：Reasoning + Acting

ReAct 是 Reasoning 和 Acting 的组合。它强调模型不要只在内部推理，也要在推理过程中穿插工具行动。

一个 ReAct 风格的循环可能是：

1. 先判断需要查资料。
2. 调用搜索工具。
3. 观察搜索结果。
4. 根据结果继续分析。
5. 必要时再次调用工具。
6. 最后生成答案。

这正是 Agent 和普通 LLM 调用的重要区别：Agent 的推理过程可以被外部观察结果不断修正。

## Action：Agent 执行的具体动作

Action 是 Agent 与环境交互的具体步骤。它可以是调用工具、执行代码、查询数据库、操作界面，也可以是向用户追问澄清信息。

从 Agent 类型看，常见实现包括：

| Agent 类型 | 描述 |
| --- | --- |
| JSON Agent | 用 JSON 格式描述要执行的动作和参数。 |
| Code Agent | 生成可被外部解释器执行的代码块。 |
| Function-calling Agent | JSON Agent 的一种常见子类，模型被训练或约束为生成函数调用消息。 |

从 Action 类型看，常见动作包括：

| Action 类型 | 描述 |
| --- | --- |
| Information Gathering | 搜索网页、查询数据库、检索文档。 |
| Tool Usage | 调用 API、运行计算、执行代码。 |
| Environment Interaction | 操作数字界面或控制物理设备。 |
| Communication | 与用户对话，或与其他 Agent 协作。 |

### Stop and Parse

许多 Agent 系统会使用 `Stop and Parse` 思路来执行动作：

1. 模型先按结构化格式生成动作，例如 JSON 或代码块。
2. 一旦动作文本生成完毕，就停止继续生成普通自然语言。
3. 外部 parser 读取这个结构化输出，解析工具名和参数，再真正调用工具。

这样做的好处是可控。模型不直接“执行世界”，而是先生成可检查的动作描述，再由运行时系统解析和执行。

### Code Agents

Code Agent 的核心想法是：与其让模型输出简单 JSON，不如让它输出可执行代码，通常是 Python 这类高级语言。

这种方式适合需要复杂中间逻辑的任务，例如数据处理、循环调用、条件判断、多步骤计算等。它的风险也更高，因为可执行代码需要更严格的沙箱、权限控制和安全审计。

## Observation：行动之后的反馈

Observation 是 Agent 感知行动结果的方式。它可能来自 API 返回值、搜索结果、错误信息、系统日志、文件内容，也可能来自用户反馈。

Observation 的作用不是简单地记录结果，而是更新 Agent 的上下文，让下一轮 Thought 更准确。

它通常包含三个过程：

- Collects Feedback：接收动作成功或失败的信息。
- Appends Results：把新信息加入上下文，相当于更新短期记忆。
- Adapts Strategy：根据反馈调整后续计划和动作。

如果工具调用失败，Observation 里可能会有错误信息；一个好的 Agent 应该能读懂错误，决定重试、换工具、修改参数，或者向用户询问更多信息。

## System Prompt：定义 Agent 运行规则

System prompt 是 Agent 的重要控制面。它通常至少包含两部分：

1. 工具信息：有哪些工具、每个工具能做什么、需要什么参数、返回什么结果。
2. 循环规则：Agent 应该如何在 Thought、Action、Observation 之间切换，什么时候继续，什么时候停止。

对于工具型 Agent，system prompt 不能只写“你很聪明”。它需要明确告诉模型：目标是什么、工具怎么用、输出格式是什么、遇到错误如何处理、什么时候返回最终答案。

## 最后复盘

Agent 的核心不是某个单独模型，而是一个围绕目标运行的系统。

最小理解可以压缩成：

- LLM 提供理解、生成、推理和规划能力。
- Tools 提供与外部环境交互的能力。
- MCP 试图标准化工具接入方式。
- Thought、Action、Observation 构成 Agent 的基本运行循环。
- System prompt 定义工具说明和循环规则。

当这些部分被组织起来，模型就不只是回答问题，而是在用户目标下持续观察、决策和行动。
