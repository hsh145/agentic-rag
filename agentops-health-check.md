# Agentic RAG — AgentOps 健康检查报告

检查日期: 2026-07-18
检查工具: agentops-awesome-list
选择难度: **T3 Production Project**

## 分级说明

- ✅ **present**: 完整实现
- ⚠️ **weak**: 部分实现，有缺口
- ❌ **missing**: 无证据找到
- ➖ **intentionally-not-needed**: 当前难度不需要

T3 要求: R=必须, L=轻量, O=可选

---

## 1. Boundary（边界层）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **系统边界** | R | ⚠️ weak | README 写了多格式RAG，但没有明确的 non-goals | 未定义系统不做什么（如实时对话、代码执行） |
| **任务接收** | R | ✅ present | `parse_intent` 节点分类意图、检测支持/不支持的请求 | — |
| **身份/会话** | R | ✅ present | `session_id` + 多轮对话 + 记忆 | — |

## 2. Runtime Core（运行时核心）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **Agent 循环** | R | ✅ present | LangGraph 7节点 + 2条件边，完整 observe/think/act | — |
| **规划器** | R | ⚠️ weak | `plan_retrieval` 拆子查询，但没有动态 plan revision | 第一次规划失败后没有回退策略 |
| **路由** | R | ✅ present | `route_after_intent` / `route_after_search` 条件边 | — |
| **执行器** | R | ✅ present | `execute_search` 带重试、超时容错 | — |
| **反射器** | R | ✅ present | `evaluate_evidence` + `reflect_search` 完整反射闭环 | — |
| **终止器** | R | ⚠️ weak | `max_iterations` 硬限制，但没有 stuck-run 检测 | 无限循环保护依赖 max_iterations，无超时熔断 |

## 3. Contracts（契约层）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **类型化消息** | R | ✅ present | TypedDict + Pydantic 模型 | — |
| **状态schema** | R | ✅ present | `AgenticRAGState` 合并/覆盖规则 | — |
| **工具schema** | R | ⚠️ weak | `DocumentParserTool` 有入参出参，但没有错误类型/幂等性 | 无工具级别错误类型、幂等性声明 |
| **产物schema** | R | ❌ missing | 无 artifact versioning | chunk 没有版本号或校验和 |
| **交接schema** | O | ➖ | 单 Agent 场景不需要 | — |

## 4. Model/Context（模型与上下文）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **模型层** | R | ✅ present | qwen-turbo/qwen-max 分层、结构化输出 | — |
| **上下文组装** | R | ✅ present | RRF 融合 + 可选 reranker + chunk 拼接 | — |

## 5. Memory（记忆系统）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **工作记忆** | R | ✅ present | LangGraph 运行时 state 对象 | — |
| **短期记忆** | R | ✅ present | SQLite SessionMemory + checkpoint | — |
| **长期记忆** | O/R | ✅ present | LLM 事实提取 + 向量召回 + 遗忘机制 | — |

## 6. Tools/Actions（工具层）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **工具层** | R | ⚠️ weak | 有 tool schema，无权限/审批/rollback | 缺少最少权限原则、人工审批、回滚 |
| **代码沙箱** | O | ➖ | 不执行代码 | — |

## 7. Project Control（项目控制）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **项目账本** | R | ❌ missing | 无项目级记录文件 | 缺少决策和变更的追踪记录 |
| **证据系统** | R | ⚠️ weak | 有 agentic_trace、有 golden set | trace 有但无 claim/evidence 映射 |
| **门控系统** | R | ❌ missing | 无 evidence gate / human gate / release gate | 高危操作无自动检查或人工审批 |

## 8. Artifacts（产物管理）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **产物管理** | R | ❌ missing | 无 artifact store | 答案、报告、chunk 无版本管理 |

## 9. Multi-Agent（多Agent）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **Agent 注册表** | O/R | ➖ | 单 Agent 系统 | — |
| **角色矩阵** | O/R | ➖ | — | — |
| **任务路由** | R | ➖ | — | — |
| **协调状态** | R | ➖ | — | — |
| **冲突仲裁** | R | ➖ | — | — |
| **交接生命周期** | O/R | ➖ | — | — |

## 10. Protocols（协议层）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **A2A 边界** | O/R | ➖ | 不跨系统通信 | — |
| **MCP 边界** | O/R | ➖ | 不使用 MCP 协议 | — |

## 11. Quality（质量保障）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **可观测性** | R | ✅ present | loguru + trace_id + agentic_trace + 耗时记录 | — |
| **评估** | R | ✅ present | 4层评估 + golden set + RAGAS | — |
| **护栏/安全** | R | ❌ missing | 无 prompt injection 检测、无 PII 过滤 | 输入无安全检查、输出无 PII 脱敏 |

## 12. Runtime Platform（运行时平台）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **部署/运行时** | R | ✅ present | FastAPI + Uvicorn，health check | — |
| **运维/runbook** | R | ❌ missing | 无 stuck-run 恢复、无 incident 响应 | 无运维手册、事故响应流程 |

## 13. Evolution（自我进化）

| 组件 | T3要求 | 状态 | 证据 | 差距 |
|---|---|---|---|---|
| **自我进化** | R with gates | ⚠️ weak | feedback API + golden set | 反馈收集了但不自动触发优化循环 |

---

## 汇总

### 总体评定: ⚠️ **Risky** — 可通过，但三个关键缺口需要上线前补齐

### 统计

| 评级 | 数量 |
|---|---|
| ✅ present | 20 |
| ⚠️ weak | 8 |
| ❌ missing | 7 |
| ➖ intentionally-not-needed | 6 |
| **总计组件** | **41**（不含 multi-agent + protocols 的 10 个 N/A） |

### P0 缺口（上线前必须修复）

| 优先级 | 组件 | 为什么重要 | 修复方案 |
|---|---|---|---|
| **P0** | 护栏/安全 | 无 prompt injection 检测，恶意输入可直接操控 | agent/nodes.py 加输入校验层，拦截 prompt injection 模式 |
| **P0** | 门控系统 | 高危操作无审批/回滚 | 写操作前检查 evidence gate，重要操作请求人工确认 |
| **P0** | 运维手册 | 出事不知道怎么办 | 写 RUNBOOK.md，定义卡住恢复、坏记忆修复、外部漂移处理 |

### P1 缺口（迭代周期内修复）

| 优先级 | 组件 | 修复方案 |
|---|---|---|
| **P1** | 规划器回退 | `plan_retrieval` LLM 调用失败时走规则兜底（当前抛异常） |
| **P1** | 终止器熔断 | 单节点超时 >60s 触发熔断，而非等 LLM 超时 |
| **P1** | 工具 schema | 给 `DocumentParserTool` 加幂等性声明和错误类型枚举 |
| **P1** | 产物管理 | 搜索结果和报告加版本号 + 校验和 |
| **P1** | 自我进化 | feedback → 自动重跑 golden set → 触发告警如果 pass rate 下降 >10% |

### P2 缺口（长期改进）

| 优先级 | 组件 |
|---|---|
| **P2** | 项目账本 | 用 `docs/` 下的决策记录追踪每次架构变更 |
| **P2** | 证据系统 | 回答中的每个 claim 绑定 source chunk id |
| **P2** | 产物 schema | chunk 和 report 的版本化存储 |
