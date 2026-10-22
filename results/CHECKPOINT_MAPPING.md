# Hugging Face checkpoint 与实验映射

权重仓库：[eexian/gemma4-31b-w4-weighted-reasoning-lora](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora)。本次按公开 revision `79d76f1f79c79aec6645d0906f4088b417c8730a` 核对，链接固定到该 revision，避免后续覆盖导致指向不同权重。共 20 个 checkpoint 目录，另有一个根目录别名；权重文件不复制到 GitHub。

## 已完成评测的对应权重

| 评测 | Hugging Face checkpoint | 成绩 | 身份校验 |
|---|---|---|---|
| [逐题结果](history-baseline-lora/lora246-20261021/) | [checkpoints/checkpoint-246](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/checkpoints/checkpoint-246) | 37/126，51 次超时 | adapter SHA256 与评测指纹一致 |
| [逐题结果](morning25-full129/) | [runs/remediation-20261022/checkpoint-25](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-25) | 51/126，38 次超时 | adapter SHA256 与评测指纹一致 |

早上 remediation checkpoint-25 与新版 prefix-v2 checkpoint-25 属于不同训练批次，权重不同。51/126 对应前者；不能将该成绩挂到新版 prefix-v2。根目录 adapter 与 `checkpoints/checkpoint-246` 的 SHA256 相同，根目录不是额外一次训练。

## 全部已上传 checkpoint

| 批次 | Step | 固定版本权重目录 | 对应代码 / 记录 | 129 题结果 |
|---|---:|---|---|---|
| original-checkpoint-series | 50 | [checkpoints/checkpoint-50](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/checkpoints/checkpoint-50) | [原始训练来源](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/blob/79d76f1f79c79aec6645d0906f4088b417c8730a/training_provenance.json) | 本归档无完整评测 |
| original-checkpoint-series | 100 | [checkpoints/checkpoint-100](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/checkpoints/checkpoint-100) | [原始训练来源](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/blob/79d76f1f79c79aec6645d0906f4088b417c8730a/training_provenance.json) | 本归档无完整评测 |
| original-checkpoint-series | 150 | [checkpoints/checkpoint-150](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/checkpoints/checkpoint-150) | [原始训练来源](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/blob/79d76f1f79c79aec6645d0906f4088b417c8730a/training_provenance.json) | 本归档无完整评测 |
| original-checkpoint-series | 200 | [checkpoints/checkpoint-200](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/checkpoints/checkpoint-200) | [原始训练来源](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/blob/79d76f1f79c79aec6645d0906f4088b417c8730a/training_provenance.json) | 本归档无完整评测 |
| original-checkpoint-series | 246 | [仓库根目录（checkpoint-246 别名）](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a) | [原始训练来源](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/blob/79d76f1f79c79aec6645d0906f4088b417c8730a/training_provenance.json) | lora246-20261021 |
| original-checkpoint-series | 246 | [checkpoints/checkpoint-246](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/checkpoints/checkpoint-246) | [原始训练来源](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/blob/79d76f1f79c79aec6645d0906f4088b417c8730a/training_provenance.json) | lora246-20261021 |
| prefix-v2-20261022 | 25 | [runs/prefix-v2-20261022/checkpoint-25](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/prefix-v2-20261022/checkpoint-25) | [training/prefix-v2](../training/prefix-v2/) | 本归档无完整评测 |
| prefix-v2-20261022 | 50 | [runs/prefix-v2-20261022/checkpoint-50](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/prefix-v2-20261022/checkpoint-50) | [training/prefix-v2](../training/prefix-v2/) | 本归档无完整评测 |
| remediation-20261022 | 5 | [runs/remediation-20261022/checkpoint-5](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-5) | [training/morning-remediation](../training/morning-remediation/) | 本归档无完整评测 |
| remediation-20261022 | 10 | [runs/remediation-20261022/checkpoint-10](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-10) | [training/morning-remediation](../training/morning-remediation/) | 本归档无完整评测 |
| remediation-20261022 | 15 | [runs/remediation-20261022/checkpoint-15](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-15) | [training/morning-remediation](../training/morning-remediation/) | 本归档无完整评测 |
| remediation-20261022 | 20 | [runs/remediation-20261022/checkpoint-20](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-20) | [training/morning-remediation](../training/morning-remediation/) | 本归档无完整评测 |
| remediation-20261022 | 25 | [runs/remediation-20261022/checkpoint-25](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-25) | [training/morning-remediation](../training/morning-remediation/) | morning25-full129-r3 |
| remediation-20261022 | 30 | [runs/remediation-20261022/checkpoint-30](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-30) | [training/morning-remediation](../training/morning-remediation/) | 本归档无完整评测 |
| remediation-20261022 | 35 | [runs/remediation-20261022/checkpoint-35](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-35) | [training/morning-remediation](../training/morning-remediation/) | 本归档无完整评测 |
| remediation-20261022 | 40 | [runs/remediation-20261022/checkpoint-40](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-40) | [training/morning-remediation](../training/morning-remediation/) | 本归档无完整评测 |
| remediation-20261022 | 45 | [runs/remediation-20261022/checkpoint-45](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-45) | [training/morning-remediation](../training/morning-remediation/) | 本归档无完整评测 |
| remediation-20261022 | 50 | [runs/remediation-20261022/checkpoint-50](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-50) | [training/morning-remediation](../training/morning-remediation/) | 本归档无完整评测 |
| remediation-20261022 | 55 | [runs/remediation-20261022/checkpoint-55](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-55) | [training/morning-remediation](../training/morning-remediation/) | 本归档无完整评测 |
| remediation-20261022 | 60 | [runs/remediation-20261022/checkpoint-60](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-60) | [training/morning-remediation](../training/morning-remediation/) | 本归档无完整评测 |
| remediation-20261022 | 65 | [runs/remediation-20261022/checkpoint-65](https://huggingface.co/eexian/gemma4-31b-w4-weighted-reasoning-lora/tree/79d76f1f79c79aec6645d0906f4088b417c8730a/runs/remediation-20261022/checkpoint-65) | [training/morning-remediation](../training/morning-remediation/) | 本归档无完整评测 |

## 使用与核验

[完整机器可读映射](huggingface-checkpoint-map.json) 保存 repo、固定 revision、subfolder、权重大小、完整 SHA256 和关联评测 ID。Hub LFS 对象 SHA256 与 checkpoint-info 中的权重哈希相互核对；已评测的两份 adapter 再与运行指纹比对，新版 step25/50 与既有上传回执核对。本次未重复下载权重。

加载时同时指定 base model、固定 revision 和对应 adapter 子目录。所有这些目录都是 LoRA adapter，不是完整 base model，也不是可直接运行的 submission.zip。相同 step 不能跨批次互换。基模 revision 为 `52f3f65bc7a02d555763bc923bd1d9094898219d`。

其余历史基模、多模型实验不对应本仓库的自训 checkpoint；模型及外部 adapter 来源见各自 configuration.json。未在本归档完成 129 题评测的 checkpoint 明确标为“无完整评测”，不继承相邻 step 的成绩。训练快照仍保留原采集时间，不代表当前训练进度。
