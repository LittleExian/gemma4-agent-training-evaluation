# 与Notebook v2的观测比较

| 方法 | 三轮通过数（分母126） | 平均正确率 | 三轮全过 | 至少一次通过 |
|---|---|---:|---:|---:|
| Teich | 57 / 51 / 54 | 42.86% | 43 | 63 |
| Notebook v2 | 51 / 55 / 47 | 40.48% | 37 | 63 |

两组至少一次通过的题目交集59题，三轮全过交集36题。

Teich独有至少一次通过：fastapi_14482, rich_3676, rich_3777, rich_4079。

v2独有至少一次通过：fastapi_14605, fastapi_14616, rich_3180, rich_3506。

Teich使用Gemma4 Teich BF16权重在线FP8、15分钟/56工具/thinking6000；v2使用官方打包W4、8分钟/28工具/thinking4096。两组全部seed42、top-k64，硬件及中断恢复过程不同。这是实际运行结果的描述，不能把差别单独归因于模型或策略。[Teich条件](README.md) · [v2条件](../notebook-v2-repeat3-20261026/README.md) · [机器可读结果](comparison-to-v2.json)。
