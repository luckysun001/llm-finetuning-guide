<div align="center">

# 🧠 LLM 微调完整指南

**从零到一，大模型微调全流程最佳实践。**<br/>
数据准备 · LoRA/QLoRA · 模型训练 · 评估

<p>
<a href="https://github.com/luckysun001/llm-finetuning-guide/actions/workflows/ci.yml"><img src="https://github.com/luckysun001/llm-finetuning-guide/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="#"><img src="https://img.shields.io/badge/Status-Beta-F97316?style=for-the-badge" alt="Status"></a>
<a href="#"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
<a href="#"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
</p>

<p>
<a href="#-核心能力"><kbd> &nbsp; ✨ 核心能力 &nbsp; </kbd></a>
<a href="#-微调方法"><kbd> &nbsp; 🎛️ 微调方法 &nbsp; </kbd></a>
<a href="#-快速开始"><kbd> &nbsp; 🚀 快速开始 &nbsp; </kbd></a>
</p>

</div>

---

## 💡 项目背景

大模型微调的痛点：

- **技术栈复杂**：涉及 PyTorch、HuggingFace、PEFT、LoRA、QLoRA 等 10+ 个库
- **坑太多**：显存不足、收敛慢、灾难性遗忘、过拟合、效果不达标
- **最佳实践分散**：没有统一的标准流程，每个人都有自己的一套
- **复现困难**：同样的数据同样的代码，换个环境结果就不一样

**本项目**：把 LLM 微调的完整流程标准化，封装成傻瓜式的工具，附带最佳实践和踩坑指南。

---

## ✨ 核心能力

### 🎛️ 微调方法

| 方法 | 状态 | 显存需求 | 说明 |
|---|---|---|---|
| **LoRA** | ✅ 已实现 | ~15% 全参 | 低秩适配，通用场景首选 |
| **QLoRA** | ✅ 已实现 | ~5% 全参 | 4-bit 量化 + LoRA，显存受限场景 |
| **全参指令微调** | ✅ 已实现 | 100% | 全参数微调示例 |
| **AdaLoRA** | 🔲 TODO | ~10% 全参 | 自适应秩分配 |
| **IA³** | 🔲 TODO | ~1% 全参 | 极快轻量微调 |

### 📊 评估体系

- **困惑度 (Perplexity)** — 语言模型质量评估（已实现）
- **自定义指标** — 支持扩展评估逻辑（已实现）

### 📦 待实现功能

以下功能在 README 中曾被提及，但尚未实现：

- ~~多格式导出（GGUF/ONNX）~~ → TODO
- ~~模型量化（4bit/8bit/16bit）~~ → TODO
- ~~vLLM 推理加速~~ → TODO
- ~~FastAPI 部署~~ → TODO
- ~~Jupyter 教程 notebooks~~ → TODO
- ~~数据集质量检查 + 去重~~ → TODO
- ~~超参数搜索~~ → TODO

---

## 📁 项目结构

```text
llm-finetuning-guide/
├── pyproject.toml              # Poetry 项目配置
├── .gitignore
├── Makefile                    # 常用命令封装
├── README.md
├── LICENSE
│
├── llm_finetuning/             # 核心代码
│   ├── __init__.py
│   ├── config.py               # 配置管理（数据类序列化/反序列化）
│   ├── engine.py               # 训练引擎（SFTTrainer 封装）
│   │
│   ├── data/                   # 数据模块
│   │   ├── __init__.py
│   │   └── processor.py        # 多格式数据加载 + 格式转换
│   │
│   ├── methods/                # 微调方法
│   │   ├── __init__.py
│   │   └── lora.py             # LoRA/QLoRA 配置生成
│   │
│   ├── evaluation/             # 评估模块
│   │   ├── __init__.py
│   │   └── evaluator.py        # 模型评估器
│   │
│   ├── visualization/          # 可视化
│   │   ├── __init__.py
│   │   └── plotter.py          # 训练曲线绘制
│   │
│   └── utils/
│       └── __init__.py
│
├── examples/                   # 示例代码
│   ├── 01_lora_basics.py       # LoRA 基础微调
│   ├── 02_qlora_advanced.py    # QLoRA 进阶微调
│   └── 03_full_instruction_tuning.py  # 全参指令微调
│
├── scripts/                    # 命令行工具
│   └── cli.py                  # CLI 入口
│
└── tests/                      # 单元测试
    ├── __init__.py
    ├── test_config.py          # 配置序列化测试
    ├── test_data.py            # 数据加载测试
    └── test_processor.py       # 数据处理器测试
```

> **注意**：README 之前版本声称的 `trainers/`（5 个文件）、`export/`、`deploy/`、`configs/`、`datasets/`、`notebooks/`、`quickstart.py`、`requirements.txt`、`.env.example`、`assets/` 等目录和文件均不存在。实际功能以本文件结构为准。

---

## 🚀 快速开始

### 环境准备

```bash
# 推荐使用 Python 3.11+
python --version

# 安装依赖（需要 Poetry）
cd 04-production-apps/llm-finetuning-guide
poetry install

# 或安装最小依赖（仅核心训练功能）
poetry install --extras minimal

# 验证 PyTorch + CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### LoRA 微调示例

```python
from llm_finetuning.config import TrainingConfig
from llm_finetuning.engine import TrainingEngine
from llm_finetuning.data.processor import DataProcessor

# 1. 准备数据
processor = DataProcessor()
dataset = processor.load("data/my_data.jsonl")

# 2. 配置训练
config = TrainingConfig(
    model_name="Qwen/Qwen-7B-Chat",
    method="qlora",
    num_epochs=3,
    batch_size=4,
)

# 3. 开始训练
engine = TrainingEngine(config)
engine.train(dataset)
```

### 命令行使用

```bash
# 微调
python scripts/cli.py train \
    --model Qwen/Qwen-7B-Chat \
    --dataset data/my_data.jsonl \
    --method qlora \
    --epochs 3
```

---

## 💻 显存需求参考

| 模型 | 全参微调 | LoRA | QLoRA 4-bit |
|---|---|---|---|
| 7B | ~90GB | ~12GB | ~4GB |
| 13B | ~180GB | ~20GB | ~8GB |
| 34B | ~400GB | ~50GB | ~20GB |
| 70B | ~800GB | ~100GB | ~40GB |

*注：实际显存取决于 batch size、序列长度、梯度累积等参数*

---

## 🧪 运行测试

```bash
# 基础测试（不需要 GPU）
pytest tests/ -v
```

---

## 📚 最佳实践

**数据准备**
- 高质量数据 > 大量数据
- 至少 1000 条高质量样本起步
- 对话数据保持格式一致

**训练配置**
- LoRA Rank：8-64 足够，不是越大越好
- 学习率：1e-4 ~ 5e-4 是黄金区间
- Epochs：2-5 轮足够，多了容易过拟合

**评估验证**
- 不要只看困惑度，一定要人工抽样检查
- 保留 10% 数据做 hold-out 测试集
- A/B 测试是黄金标准

---

## 🔗 相关项目

本仓库讲「**怎么把模型训出来**」。

如果你更关心「**不训练也能把大模型用起来**」，推荐作者的另一个开源项目：

> ### 📚 [开源 AI 实战指南（ai-education）](https://github.com/luckysun001/ai-education)
> 覆盖 8 大行业 + 7 大职业 + 无代码实战，全部是可直接复制使用的 Prompt 模板与工具选型建议。

想先把原理和代码搞懂，另见 [AI-Learning-Playground](https://github.com/luckysun001/AI-Learning-Playground)。

---

## 📝 开发计划

- [ ] 实现 `export/` 模块（GGUF/ONNX 导出 + 量化）
- [ ] 实现 `deploy/` 模块（FastAPI 服务 + vLLM 推理）
- [ ] 补充 Jupyter notebooks 教程
- [ ] 补充 `configs/` YAML 配置文件
- [ ] 补充 `datasets/` 示例数据集
- [ ] 实现数据清洗/去重模块
- [ ] 补充更多测试用例

---

## 📄 许可证

MIT License - 完全开源免费。
