# 数据集目录

本目录用于放置训练数据。**仓库的 `.gitignore` 忽略了 `*.jsonl / *.json / *.csv / *.parquet`**，
避免大文件进入版本库，因此示例数据需要本地生成：

```bash
python scripts/make_sample_data.py
# 默认生成 datasets/sample_instructions.jsonl（10 条中文指令样本）
```

## 支持的格式

| 格式 | 加载方式 | 说明 |
|---|---|---|
| JSONL | `DatasetLoader.load_from_jsonl(path)` | 每行一个样本，推荐 |
| JSON | `DatasetLoader.load_from_json(path)` | 数组形式 |
| CSV | `DatasetLoader.load_from_csv(path)` | 返回 DataFrame |
| Parquet | `DatasetLoader.load_from_parquet(path)` | 大样本更高效 |
| HuggingFace Hub | `DatasetLoader.load_from_hub(name)` | 需要 `datasets` |

## 指令微调样本字段

```json
{"instruction": "任务描述", "input": "可选的补充输入", "output": "期望输出"}
```

`input` 可为空字符串（纯指令场景）。字段缺失或文本为空的样本会被
`DataCleaner` 直接剔除，训练前建议先跑一遍清洗：

```python
from llm_finetuning import DataCleaner

cleaned, report = DataCleaner(min_length=10, fuzzy_dedup=True).clean(records)
print(report.summary())
```
