# Notebook v2 三轮结果与逐题差异

全部三轮 `seed=42`，每轮129题，其中126题可评；相同3题预先排除。通过与超时是独立字段，不能视为互斥。

| 轮次 | 通过 / 可评 | 正确率 | 超时 | 超时但通过 | Agent平均秒数 |
|---|---:|---:|---:|---:|---:|
| 1 | 51/126 | 40.48% | 18 | 0 | 335.23 |
| 2 | 55/126 | 43.65% | 14 | 0 | 315.11 |
| 3 | 47/126 | 37.30% | 34 | 0 | 348.59 |

三轮平均正确率 **40.48%**；最高与最低相差 **6.35个百分点**。

**37题三轮全通过，63题三轮全失败，26题通过与否发生变化；63题至少通过一次。**

## 三轮通过模式

`1`为通过，`0`为未通过，位序为第1、2、3轮。

| 模式 | 题数 |
|---|---:|
| `000` | 63 |
| `001` | 2 |
| `010` | 4 |
| `011` | 6 |
| `100` | 4 |
| `101` | 2 |
| `110` | 8 |
| `111` | 37 |

## 两两比较

| 从→到 | 新增通过 | 失去通过 | 结果不同题数 | 通过集合交集/并集 | Jaccard |
|---|---:|---:|---:|---:|---:|
| 1→2 | 10 | 6 | 16/126 | 45/61 | 0.7377 |
| 1→3 | 8 | 12 | 20/126 | 39/59 | 0.6610 |
| 2→3 | 4 | 12 | 16/126 | 43/59 | 0.7288 |

## 哪些题发生变化

| 题目 | 模式 | 第1轮 | 第2轮 | 第3轮 |
|---|---|---|---|---|
| `fastapi_12942` | `110` | [通过](round-1/traces/fastapi_12942.json) | [通过](round-2/traces/fastapi_12942.json) | [失败（超时）](round-3/traces/fastapi_12942.json) |
| `fastapi_14349` | `011` | [失败](round-1/traces/fastapi_14349.json) | [通过](round-2/traces/fastapi_14349.json) | [通过](round-3/traces/fastapi_14349.json) |
| `fastapi_14419` | `010` | [失败](round-1/traces/fastapi_14419.json) | [通过](round-2/traces/fastapi_14419.json) | [失败（超时）](round-3/traces/fastapi_14419.json) |
| `fastapi_14430` | `011` | [失败](round-1/traces/fastapi_14430.json) | [通过](round-2/traces/fastapi_14430.json) | [通过](round-3/traces/fastapi_14430.json) |
| `fastapi_14459` | `110` | [通过](round-1/traces/fastapi_14459.json) | [通过](round-2/traces/fastapi_14459.json) | [失败（超时）](round-3/traces/fastapi_14459.json) |
| `fastapi_14485` | `100` | [通过](round-1/traces/fastapi_14485.json) | [失败](round-2/traces/fastapi_14485.json) | [失败（超时）](round-3/traces/fastapi_14485.json) |
| `fastapi_14605` | `010` | [失败](round-1/traces/fastapi_14605.json) | [通过](round-2/traces/fastapi_14605.json) | [失败](round-3/traces/fastapi_14605.json) |
| `fastapi_14616` | `010` | [失败](round-1/traces/fastapi_14616.json) | [通过](round-2/traces/fastapi_14616.json) | [失败](round-3/traces/fastapi_14616.json) |
| `fastapi_14791` | `011` | [失败](round-1/traces/fastapi_14791.json) | [通过](round-2/traces/fastapi_14791.json) | [通过](round-3/traces/fastapi_14791.json) |
| `fastapi_14873` | `110` | [通过](round-1/traces/fastapi_14873.json) | [通过](round-2/traces/fastapi_14873.json) | [失败（超时）](round-3/traces/fastapi_14873.json) |
| `fastapi_15589` | `110` | [通过](round-1/traces/fastapi_15589.json) | [通过](round-2/traces/fastapi_15589.json) | [失败（超时）](round-3/traces/fastapi_15589.json) |
| `fastapi_5624` | `001` | [失败](round-1/traces/fastapi_5624.json) | [失败](round-2/traces/fastapi_5624.json) | [通过](round-3/traces/fastapi_5624.json) |
| `fastapi_9425` | `110` | [通过](round-1/traces/fastapi_9425.json) | [通过](round-2/traces/fastapi_9425.json) | [失败](round-3/traces/fastapi_9425.json) |
| `fastapi_9753` | `010` | [失败](round-1/traces/fastapi_9753.json) | [通过](round-2/traces/fastapi_9753.json) | [失败（超时）](round-3/traces/fastapi_9753.json) |
| `requests_7328` | `110` | [通过](round-1/traces/requests_7328.json) | [通过](round-2/traces/requests_7328.json) | [失败（超时）](round-3/traces/requests_7328.json) |
| `requests_7427` | `101` | [通过](round-1/traces/requests_7427.json) | [失败（超时）](round-2/traces/requests_7427.json) | [通过](round-3/traces/requests_7427.json) |
| `requests_7505` | `110` | [通过](round-1/traces/requests_7505.json) | [通过](round-2/traces/requests_7505.json) | [失败（超时）](round-3/traces/requests_7505.json) |
| `rich_2943` | `011` | [失败（超时）](round-1/traces/rich_2943.json) | [通过](round-2/traces/rich_2943.json) | [通过](round-3/traces/rich_2943.json) |
| `rich_3043` | `011` | [失败](round-1/traces/rich_3043.json) | [通过](round-2/traces/rich_3043.json) | [通过](round-3/traces/rich_3043.json) |
| `rich_3180` | `011` | [失败（超时）](round-1/traces/rich_3180.json) | [通过](round-2/traces/rich_3180.json) | [通过](round-3/traces/rich_3180.json) |
| `rich_3278` | `100` | [通过](round-1/traces/rich_3278.json) | [失败（超时）](round-2/traces/rich_3278.json) | [失败（超时）](round-3/traces/rich_3278.json) |
| `rich_3480` | `100` | [通过](round-1/traces/rich_3480.json) | [失败（超时）](round-2/traces/rich_3480.json) | [失败（超时）](round-3/traces/rich_3480.json) |
| `rich_3506` | `110` | [通过](round-1/traces/rich_3506.json) | [通过](round-2/traces/rich_3506.json) | [失败](round-3/traces/rich_3506.json) |
| `rich_3521` | `100` | [通过](round-1/traces/rich_3521.json) | [失败](round-2/traces/rich_3521.json) | [失败（超时）](round-3/traces/rich_3521.json) |
| `rich_3718` | `001` | [失败](round-1/traces/rich_3718.json) | [失败](round-2/traces/rich_3718.json) | [通过](round-3/traces/rich_3718.json) |
| `rich_3894` | `101` | [通过](round-1/traces/rich_3894.json) | [失败](round-2/traces/rich_3894.json) | [通过](round-3/traces/rich_3894.json) |

## 按项目

| 项目 | 可评题数 | 第1轮通过 | 第2轮通过 | 第3轮通过 | 结果变化题数 |
|---|---:|---:|---:|---:|---:|
| fastapi | 65 | 23 | 29 | 21 | 14 |
| httpx | 1 | 1 | 1 | 1 | 0 |
| requests | 12 | 12 | 11 | 10 | 3 |
| rich | 48 | 15 | 14 | 15 | 9 |

## 对照口径和文件

- 第1、2轮：GPU3单卡，两路任务；第1轮经历中断恢复，跨两次模型服务。
- 第3轮：GPU2、GPU3两卡，每卡两路，共四路。模型、提示词、预算及seed相同；硬件及调度条件不完全一致。
- 不能将差异单独归因于随机种子、硬件或某项策略；这里报告实际运行的重复一致性。
- `task-results-long.csv`：全部387条任务/轮次记录；排除项明确标记为 `not_scored`。
- `task-comparison.csv`：129题并排指标及原始补丁SHA；`changed-tasks.csv`：其中26题正确性变化项。
- `comparison.json`：全部通过模式的题目ID、两两新增/失去通过的ID、项目统计。
- 补丁SHA比较的是原始生成字节；脱敏公开补丁字节的SHA另由发布清单记录。
- `24`题三轮补丁原始字节完全一致；补丁不同不代表正确性必然不同。
- 上述相同补丁中，7题三轮均为空补丁、17题为非空补丁；JSON分别列出题目ID。
- 两两比较中未发现“原始补丁相同但通过结果不同”的题；可在 JSON 的 `same_patch_different_correctness_ids` 核对。这不排除不同补丁在语义上等价。
- 预先排除：`fastapi_14077`、`fastapi_14851`（空补丁也通过）；`requests_6757`（环境校准异常）。
