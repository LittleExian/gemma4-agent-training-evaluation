# Gemma 4 Agent — 训练与评测记录

本仓库保存 Gemma 4 31B W4A16 + LoRA 的运行代码、环境配置、预检证据和结果快照。来源为 [TheWayLost/gemmacodeagent](https://github.com/TheWayLost/gemmacodeagent) 及本地适配，固定提交与原始文件哈希见 [来源说明](provenance/UPSTREAM.md)。

## 历史全量评测

已整理 **13 个完整的 129 题配置**：早期 budget 两轮、校准基线、七模型对比、Teich top-k64 重跑、LoRA246，以及 morning25。详见 [历史汇总表](results/HISTORICAL_129.md)、[参数 CSV](results/historical-129-index.csv) 和各组逐题结果。不同预算和环境的成绩分开注明，不视为受控排行榜。

## 当前实验

| 实验 | 状态 / 结果 | 对应文件 |
|---|---|---|
| 早上 remediation 训练 checkpoint-25 的 129 题评测 | 已完成：**51/126（40.48%）**，38 道达到时间预算；3 道已知异常单列 | [逐题 CSV](results/morning25-full129/task-results.csv)、[完整结构化结果](results/morning25-full129/task-results.json) |
| 新版 deployment-prefix-v2 双卡训练 | **运行中快照：247/356 步**，采集时间北京时间 2026-10-22 21:59；最终训练结果尚未包含 | [进度](results/training-prefix-v2/progress.json)、[rank0 指标](results/training-prefix-v2/metrics.rank0.jsonl)、[采集元数据](results/collection-metadata.json) |
| 旧 morning remediation 训练 | 按用户指示停止，保留 checkpoint-175 | [停止记录](results/training-morning-remediation-stopped/training-stop.json) |

51/126 的成绩对应 **早上 remediation checkpoint-25**。新版 prefix-v2 checkpoint-25 / checkpoint-50 的权重另行保存，下表给出各自地址；新版 adapter 的 129 题质量结果尚未纳入此快照。

### 129 题结果

| 项目 | 数值 |
|---|---:|
| 总题数 | 129 |
| 可评题数 | 126 |
| 解决题数 | 51 |
| 可评题正确率 | 40.48% |
| 按全部 129 题计的解决率 | 39.53% |
| 达到 agent 时间预算 | 38 |
| 已知异常单列 | 3 |

超时数与解决数是分别统计的字段，不应未经逐题核对就视为互斥类别。异常题为 `requests_6757`（环境校准异常）、`fastapi_14077` 和 `fastapi_14851`（空补丁也通过）。异常题不计为解决。

结果由最终有效的 `full129-single-gpu0-r3` 运行生成。此前 r1/r2 的提交打包失败不计入模型成绩。有效提交目录在最终版本中使用真实文件，官方编译器的符号链接拒绝检查保留，显式 sandbox 启动错误作为基础设施错误处理。

## 运行配置

- **推理**：一张 RTX PRO 6000 96GB、TP1、4 路并发、vLLM 显存利用率上限 0.65。原有其他 GPU 进程被保留。
- **模型**：`google/gemma-4-31B-it-qat-w4a16-ct`，固定 revision `52f3f65bc7a02d555763bc923bd1d9094898219d`；原生 rank16 LoRA，无权重合并或格式转换。
- **预算**：每题15分钟、56次工具调用、160轮、thinking6000、最大输出8192、temperature0.2、top-p0.95、top-k64；工具超时240秒、验证超时300秒。
- **上下文**：32768；14336 tokens 触发历史压缩；明确固定 tokenizer、chat template 和 `--chat-template-content-format string`。
- **评测运行时**：Python3.12.4、vLLM0.19.1、PyTorch2.10.0+cu128、Transformers5.13.1、Google ADK1.36.1、swegemma0.2.11、adk-submission0.2.13、LiteLLM1.83.14。
- **题目依赖**：22套按题校准的环境，Python3.10.22或3.12.4；实际 `pip freeze` 必须匹配参考校准记录。每题重新执行 gold/empty 对照后进入模型评测。
- **任务隔离**：subprocess + Linux Landlock 文件权限和原生工具路径守卫；没有容器级网络、CPU或内存配额。这是本地主机适配，不是官方沙箱/硬件完全一致性的声明。
- **新版训练**：两张 PRO6000、microbatch1/卡、accum4、global batch8、task_mean、LR2e-5、rank16、seed42、1epoch/356步、每5步保存。旧 adapter 和 token cache 不续用。

[完整环境配置](environments/environment-config-20261022.json) 包含322个评测依赖版本、22套题目 profile 和任务映射。[评测依赖锁定列表](environments/evaluation-runtime-pinned.txt) 与 [按题环境锁定列表](environments/test-profiles/) 可供核对。训练依赖以各自训练目录中的 pinned requirements 为准，不与推理环境混装。

## 代码布局

```text
training/prefix-v2/             新版训练原始实现、测试和说明
training/morning-remediation/  早上训练及 streamed CE 适配
orchestration/training/        实际训练流水线
orchestration/scripts/         本地启动、预检、服务、校准和测试脚本
evaluation/src/               最终129题评测本地适配与单元测试
evaluation/experiments/       固定版本的distributed实验runner
evaluation/submission/        评测配置与提示词；adapter权重通过HF获取
results/                       逐题结果、生成补丁、训练指标及预检证据
environments/                  运行时依赖和22套按题依赖
provenance/                    来源及原始/公开字节哈希
```

参见 [复现步骤与路径说明](REPRODUCING.md)。脚本保留执行时的目录布局，将个人路径泛化为 `/workspace/`、服务器地址替换为 `SERVER_HOST`；运行前需配置自己的路径、数据和凭据。历史机器上的哈希与当前公开文件哈希分别记录，不将脱敏后的文件冒充原始执行字节。

## 权重

权重、优化器状态和数据包不放入 Git。

| Adapter | Hugging Face |
|---|---|
| 早上 remediation checkpoint-25（本次51/126） | [runs/remediation-20261022/checkpoint-25](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/main/runs/remediation-20261022/checkpoint-25) |
| 新版 prefix-v2 checkpoint-25 | [runs/prefix-v2-20261022/checkpoint-25](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/main/runs/prefix-v2-20261022/checkpoint-25) |
| 新版 prefix-v2 checkpoint-50 | [runs/prefix-v2-20261022/checkpoint-50](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/main/runs/prefix-v2-20261022/checkpoint-50) |

## 公开范围

包含运行代码、配置、逐题指标、模型生成的候选补丁、来源/预算哈希及训练指标。原始题目包、reference/gold/test补丁、完整LLM消息/trace、凭据、W&B原始日志、模型和优化器二进制均不公开；受限原始日志保留文件大小及SHA清单用于核对。生成补丁是模型候选输出，不是参考答案，也不保证正确。

本仓库是有时间戳的结果快照，不会在服务器训练推进时自动更新。环境与源码哈希、校准控制和质量结果分别记录；训练 smoke 通过不代表修复能力提高。

依赖审计对历史评测锁文件报告了已知漏洞条目，部分本地/竞赛包无法匹配审计服务；详见 [审计范围与限制](provenance/DEPENDENCY_AUDIT.md)。保留历史版本用于结果溯源，不将其描述为无漏洞部署方案。
