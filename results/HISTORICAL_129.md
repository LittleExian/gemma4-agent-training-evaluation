# 历史 129 题评测汇总

已核对 **13 个完整模型/运行配置，共 1677 条逐题记录**，包含此前已发布的 morning25。七模型对比计 7 个配置，属于同一个实验批次；小样本、重试和失败打包不重复计数。日期型目录名沿用原运行标签，实际起止时间见各组 summary。

[完整参数 JSON](historical-129-index.json) · [汇总 CSV](historical-129-index.csv)

| 配置 | 解决题数 | 超时 | 分钟 / 工具 / 轮数 | thinking | top-k | 总并发 | 证据 |
|---|---:|---:|---|---:|---|---:|---|
| gemma4-budget-full-20261015 | 43/129 | 未统一重算 | 8 / 28 / 80 | 未知 | 未明确记录 / 见配置 | 4 | [结果与配置](history-budget/gemma4-budget-full-20261015/) |
| gemma4-budget-full-20261016-repeat1 | 40/129 | 未统一重算 | 8 / 28 / 80 | 未知 | 未明确记录 / 见配置 | 4 | [结果与配置](history-budget/gemma4-budget-full-20261016-repeat1/) |
| baseline-20261016 | 51/126 | 9 | 8 / 28 / 80 | 6000 | 未明确记录 / 见配置 | 12 | [结果与配置](history-baseline-lora/baseline-20261016/) |
| lora246-20261021 | 37/126 | 51 | 15 / 56 / 160 | 6000 | 64 | 8 | [结果与配置](history-baseline-lora/lora246-20261021/) |
| repaired-mm--gemma4_capybara | 22/126 | 83 | 15 / 56 / 160 | 6000 | 未明确记录 / 见配置 | 12 | [结果与配置](history-multimodel/repaired-mm--gemma4_capybara/) |
| repaired-mm--gemma4_kaggle | 53/126 | 13 | 15 / 56 / 160 | 6000 | 64 | 12 | [结果与配置](history-multimodel/repaired-mm--gemma4_kaggle/) |
| repaired-mm--gemma4_teich_agent | 53/126 | 34 | 15 / 56 / 160 | 6000 | 未明确记录 / 见配置 | 12 | [结果与配置](history-multimodel/repaired-mm--gemma4_teich_agent/) |
| repaired-mm--qwen35_base | 37/126 | 57 | 15 / 56 / 160 | 6000 | 未明确记录 / 见配置 | 12 | [结果与配置](history-multimodel/repaired-mm--qwen35_base/) |
| repaired-mm--qwen35_swegym | 23/126 | 68 | 15 / 56 / 160 | 6000 | 未明确记录 / 见配置 | 12 | [结果与配置](history-multimodel/repaired-mm--qwen35_swegym/) |
| repaired-mm--qwen38_base | 40/126 | 77 | 15 / 56 / 160 | 6000 | 未明确记录 / 见配置 | 12 | [结果与配置](history-multimodel/repaired-mm--qwen38_base/) |
| repaired-mm--qwen38_coder390 | 40/126 | 70 | 15 / 56 / 160 | 6000 | 未明确记录 / 见配置 | 12 | [结果与配置](history-multimodel/repaired-mm--qwen38_coder390/) |
| teich-topk64--gemma4_teich_agent | 55/126 | 24 | 15 / 56 / 160 | 6000 | 64 | 12 | [结果与配置](history-multimodel/teich-topk64--gemma4_teich_agent/) |
| morning25-full129-r3 | 51/126 | 38 | 15 / 56 / 160 | 6000 | 64 | 4 | [结果与配置](morning25-full129/) |

## 比较限制

- 前两轮 budget 的 43/129、40/129 是未经逐题对照校准的历史原始结果，存在依赖和诊断分类问题。不得改成 126 分母，也不能把差异仅解释为采样随机性。详见 [两轮审计](history-budget/paired-repeat-audit.json)。
- 其余配置均覆盖 129 个 ID，126 题纳入成绩，3 题异常单列。超时字段与解决字段分别统计，不应当作互斥类别相加。
- baseline 使用 8 分钟 / 28 工具；后续对比和 LoRA 使用 15 分钟 / 56 工具。模型、adapter、模板、服务默认值及并发也存在差异，这不是只改变 adapter 的受控实验。
- 原七模型实验的 Teich/Qwen/Capybara 请求未显式给出 top-k，服务使用 `--generation-config vllm`；不将它们冒记为 top-k64。Kaggle 基模继承模型 generation_config 的 64；独立 Teich 重跑显式配置 64。完整启动和采样证据在各组 configuration.json。
- 原实验的 6 组 pilot16 被修复实验复用；修复实验的全量题目重新运行。原/修复目录不是两轮完整七模型结果。见 [实验沿革](history-multimodel/index.json)。

## 文件范围

每组包括逐题 JSON/CSV、成绩和预算、候选补丁、原始证据文件的 SHA256/大小，以及对应代码/配置或哈希一致的既有文件引用。代码快照位于 [history-sources](../history-sources/)。候选补丁是模型生成内容；gold/test 补丁、原始题目、完整消息/trace、权重和凭据不公开。原始字节哈希与脱敏后公开文件哈希分别保留。

这是历史归档，未重新运行 GPU 评测。原有 [依赖审计](../provenance/DEPENDENCY_AUDIT.md) 只覆盖原评测锁文件，不覆盖本次新增的全部历史运行环境。
