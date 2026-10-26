# Teich 三轮结果与逐题差异

模型为 `TeichAI/Gemma-4-31B-Fable-5-Agent-Distill`，磁盘 BF16 权重、vLLM 在线 FP8，无 LoRA。全部三轮 `seed=42`、top-k64、thinking6000；每题15分钟、56次工具调用、160轮。

每轮129题，其中126题可评；相同3题预先排除。通过与超时是独立字段，不能视为互斥。上下文超限单独统计，不计为时间预算超时。

| 轮次 | 通过 / 可评 | 正确率 | 超时 | 超时但通过 | 上下文超限 | Agent平均秒数 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 57/126 | 45.24% | 31 | 0 | 1 | 581.27 |
| 2 | 51/126 | 40.48% | 29 | 0 | 0 | 553.05 |
| 3 | 54/126 | 42.86% | 27 | 0 | 0 | 580.40 |

三轮平均正确率 **42.86%**；最高与最低相差 **4.76个百分点**。

**43题三轮全通过，63题三轮全失败，20题通过与否发生变化；63题至少通过一次。**

## 三轮通过模式

`1`为通过，`0`为未通过，位序为第1、2、3轮。

| 模式 | 题数 |
|---|---:|
| `000` | 63 |
| `001` | 2 |
| `010` | 1 |
| `011` | 3 |
| `100` | 4 |
| `101` | 6 |
| `110` | 4 |
| `111` | 43 |

## 两两比较

| 从→到 | 新增通过 | 失去通过 | 结果不同题数 | 通过集合交集/并集 | Jaccard |
|---|---:|---:|---:|---:|---:|
| 1→2 | 4 | 10 | 14/126 | 47/61 | 0.7705 |
| 1→3 | 5 | 8 | 13/126 | 49/62 | 0.7903 |
| 2→3 | 8 | 5 | 13/126 | 46/59 | 0.7797 |

## 哪些题发生变化

| 题目 | 模式 | 第1轮 | 第2轮 | 第3轮 |
|---|---|---|---|---|
| `fastapi_12942` | `101` | [通过](round-1/traces/fastapi_12942.json) | [失败](round-2/traces/fastapi_12942.json) | [通过](round-3/traces/fastapi_12942.json) |
| `fastapi_14349` | `101` | [通过](round-1/traces/fastapi_14349.json) | [失败（超时）](round-2/traces/fastapi_14349.json) | [通过](round-3/traces/fastapi_14349.json) |
| `fastapi_14430` | `010` | [失败（超时）](round-1/traces/fastapi_14430.json) | [通过](round-2/traces/fastapi_14430.json) | [失败（超时）](round-3/traces/fastapi_14430.json) |
| `fastapi_14459` | `110` | [通过](round-1/traces/fastapi_14459.json) | [通过](round-2/traces/fastapi_14459.json) | [失败（超时）](round-3/traces/fastapi_14459.json) |
| `fastapi_14482` | `110` | [通过](round-1/traces/fastapi_14482.json) | [通过](round-2/traces/fastapi_14482.json) | [失败（超时）](round-3/traces/fastapi_14482.json) |
| `fastapi_14485` | `011` | [失败（超时）](round-1/traces/fastapi_14485.json) | [通过](round-2/traces/fastapi_14485.json) | [通过](round-3/traces/fastapi_14485.json) |
| `fastapi_14791` | `011` | [失败](round-1/traces/fastapi_14791.json) | [通过](round-2/traces/fastapi_14791.json) | [通过](round-3/traces/fastapi_14791.json) |
| `fastapi_14873` | `101` | [通过](round-1/traces/fastapi_14873.json) | [失败（超时）](round-2/traces/fastapi_14873.json) | [通过](round-3/traces/fastapi_14873.json) |
| `fastapi_15589` | `001` | [失败（超时）](round-1/traces/fastapi_15589.json) | [失败（超时）](round-2/traces/fastapi_15589.json) | [通过](round-3/traces/fastapi_15589.json) |
| `fastapi_5624` | `101` | [通过](round-1/traces/fastapi_5624.json) | [失败](round-2/traces/fastapi_5624.json) | [通过](round-3/traces/fastapi_5624.json) |
| `requests_7328` | `100` | [通过](round-1/traces/requests_7328.json) | [失败](round-2/traces/requests_7328.json) | [失败（超时）](round-3/traces/requests_7328.json) |
| `requests_7505` | `100` | [通过](round-1/traces/requests_7505.json) | [失败（超时）](round-2/traces/requests_7505.json) | [失败（超时）](round-3/traces/requests_7505.json) |
| `rich_3043` | `101` | [通过](round-1/traces/rich_3043.json) | [失败](round-2/traces/rich_3043.json) | [通过](round-3/traces/rich_3043.json) |
| `rich_3130` | `110` | [通过](round-1/traces/rich_3130.json) | [通过](round-2/traces/rich_3130.json) | [失败（超时）](round-3/traces/rich_3130.json) |
| `rich_3278` | `100` | [通过](round-1/traces/rich_3278.json) | [失败](round-2/traces/rich_3278.json) | [失败](round-3/traces/rich_3278.json) |
| `rich_3480` | `011` | [失败](round-1/traces/rich_3480.json) | [通过](round-2/traces/rich_3480.json) | [通过](round-3/traces/rich_3480.json) |
| `rich_3521` | `100` | [通过](round-1/traces/rich_3521.json) | [失败（超时）](round-2/traces/rich_3521.json) | [失败（超时）](round-3/traces/rich_3521.json) |
| `rich_3676` | `110` | [通过](round-1/traces/rich_3676.json) | [通过](round-2/traces/rich_3676.json) | [失败](round-3/traces/rich_3676.json) |
| `rich_3777` | `101` | [通过](round-1/traces/rich_3777.json) | [失败（超时）](round-2/traces/rich_3777.json) | [通过](round-3/traces/rich_3777.json) |
| `rich_4079` | `001` | [失败](round-1/traces/rich_4079.json) | [失败](round-2/traces/rich_4079.json) | [通过](round-3/traces/rich_4079.json) |

## 按项目

| 项目 | 可评题数 | 第1轮通过 | 第2轮通过 | 第3轮通过 | 结果变化题数 |
|---|---:|---:|---:|---:|---:|
| fastapi | 65 | 26 | 25 | 27 | 10 |
| httpx | 1 | 1 | 1 | 1 | 0 |
| requests | 12 | 12 | 10 | 10 | 2 |
| rich | 48 | 18 | 15 | 16 | 8 |

## 对照口径和文件

- 第1轮先使用GPU2、GPU3每卡两路，共四路；保留101道已完成的模型结果。恢复后改用GPU0、GPU2、GPU3每卡四路，共12路，生成剩余24题。
- 第1轮 [rich_3782](round-1/traces/rich_3782.json) 的原生成在上下文超限后保留：17次原LLM调用、空补丁；离线官方验证为24项通过、1项目标测试失败，没有重新调用模型。它作为失败结果计入126题，错误码为 `agent_context_limit`。
- 第2、3轮完整使用GPU0、GPU2、GPU3每卡四路，共12路。模型、提示词、预算及seed相同；第1轮硬件、调度和中断条件不同。
- 第1轮 `full_seconds` 对恢复题保留原开始/结束时间，不包含后来离线补丁验证的墙钟间隔；`agent_seconds` 保留原生成耗时。
- 不能将差异单独归因于随机种子、硬件或某项策略；这里报告实际运行的重复一致性。
- `task-results-long.csv`：全部387条任务/轮次记录；排除项明确标记为 `not_scored`。
- `task-comparison.csv`：129题并排指标及原始补丁SHA；`changed-tasks.csv`：其中20题正确性变化项。
- `comparison.json`：全部通过模式的题目ID、两两新增/失去通过的ID、项目统计。
- 补丁SHA比较的是原始生成字节；脱敏公开补丁字节的SHA另由发布清单记录。
- `13`题三轮补丁原始字节完全一致；补丁不同不代表正确性必然不同。
- 上述相同补丁中，0题三轮均为空补丁、13题为非空补丁；JSON分别列出题目ID。
- 两两比较中未发现“原始补丁相同但通过结果不同”的题；可在 JSON 的 `same_patch_different_correctness_ids` 核对。这不排除不同补丁在语义上等价。
- 预先排除：`fastapi_14077`、`fastapi_14851`（空补丁也通过）；`requests_6757`（环境校准异常）。
