"""
数据清洗与去重模块
==================

微调效果的天花板往往由数据质量决定。本模块提供训练前的低成本体检：

- 去掉空样本与字段缺失样本
- 归一化空白字符（全角空格、连续空行、首尾空白）
- 按内容哈希精确去重
- 可选：按归一化后的文本做「模糊去重」（去掉仅差标点/空白的样本）
- 按长度过滤过短或过长的样本

纯 Python 实现，不需要额外依赖。
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: 各类空白字符（含全角空格）统一折叠为一个半角空格
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class CleanReport:
    """清洗报告"""

    total: int = 0
    kept: int = 0
    removed_empty: int = 0
    removed_missing_field: int = 0
    removed_duplicate: int = 0
    removed_short: int = 0
    removed_long: int = 0
    removed_samples: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def removed(self) -> int:
        return self.total - self.kept

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "kept": self.kept,
            "removed": self.removed,
            "removed_empty": self.removed_empty,
            "removed_missing_field": self.removed_missing_field,
            "removed_duplicate": self.removed_duplicate,
            "removed_short": self.removed_short,
            "removed_long": self.removed_long,
        }

    def summary(self) -> str:
        """人类可读的清洗摘要"""
        return (
            f"总计 {self.total} 条，保留 {self.kept} 条，剔除 {self.removed} 条"
            f"（空 {self.removed_empty} / 缺字段 {self.removed_missing_field} / "
            f"重复 {self.removed_duplicate} / 过短 {self.removed_short} / "
            f"过长 {self.removed_long}）"
        )


class DataCleaner:
    """数据清洗器"""

    def __init__(
        self,
        text_field: str = "text",
        min_length: int = 1,
        max_length: Optional[int] = None,
        deduplicate: bool = True,
        fuzzy_dedup: bool = False,
        normalize_whitespace: bool = True,
        lowercase: bool = False,
        keep_report_samples: int = 0,
    ):
        if min_length < 0:
            raise ValueError(f"min_length 不能为负: {min_length}")
        if max_length is not None and max_length < min_length:
            raise ValueError(
                f"max_length({max_length}) 不能小于 min_length({min_length})"
            )

        self.text_field = text_field
        self.min_length = min_length
        self.max_length = max_length
        self.deduplicate = deduplicate
        self.fuzzy_dedup = fuzzy_dedup
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        self.keep_report_samples = keep_report_samples

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def clean(
        self, records: Iterable[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], CleanReport]:
        """清洗数据集

        Args:
            records: 原始样本列表（每条为 dict）

        Returns:
            (清洗后的样本, 清洗报告)
        """
        report = CleanReport()
        cleaned: List[Dict[str, Any]] = []
        seen: set = set()

        for raw in records:
            report.total += 1

            if not isinstance(raw, dict) or self.text_field not in raw:
                report.removed_missing_field += 1
                self._record_removed(report, raw, "missing_field")
                continue

            record = dict(raw)
            text = self._normalize(str(record[self.text_field]))

            if not text:
                report.removed_empty += 1
                self._record_removed(report, record, "empty")
                continue

            if len(text) < self.min_length:
                report.removed_short += 1
                self._record_removed(report, record, "too_short")
                continue

            if self.max_length is not None and len(text) > self.max_length:
                report.removed_long += 1
                self._record_removed(report, record, "too_long")
                continue

            if self.deduplicate:
                key = self._dedup_key(text)
                if key in seen:
                    report.removed_duplicate += 1
                    self._record_removed(report, record, "duplicate")
                    continue
                seen.add(key)

            record[self.text_field] = text
            cleaned.append(record)

        report.kept = len(cleaned)
        return cleaned, report

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        """归一化文本：全角转半角 + 空白折叠 + 可选小写"""
        text = unicodedata.normalize("NFKC", text)
        if self.normalize_whitespace:
            text = _WHITESPACE_RE.sub(" ", text).strip()
        if self.lowercase:
            text = text.lower()
        return text

    def _dedup_key(self, text: str) -> str:
        """去重键：精确去重用原文，模糊去重去掉标点与空白"""
        if not self.fuzzy_dedup:
            return text

        stripped = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
        return stripped

    def _record_removed(
        self, report: CleanReport, record: Any, reason: str
    ) -> None:
        if self.keep_report_samples <= 0:
            return
        if len(report.removed_samples) >= self.keep_report_samples:
            return
        if isinstance(record, dict):
            sample = dict(record)
            sample["_removed_reason"] = reason
            report.removed_samples.append(sample)


def deduplicate(records: Iterable[Dict[str, Any]], text_field: str = "text") -> List[Dict[str, Any]]:
    """只做去重的便捷函数"""
    cleaned, _ = DataCleaner(text_field=text_field).clean(records)
    return cleaned


def length_stats(records: Iterable[Dict[str, Any]], text_field: str = "text") -> Dict[str, float]:
    """统计文本长度分布（用于决定 min_length / max_length 阈值）"""
    lengths = [
        len(str(r.get(text_field, ""))) for r in records if isinstance(r, dict)
    ]
    if not lengths:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    ordered = sorted(lengths)
    mid = len(ordered) // 2
    median = (
        ordered[mid]
        if len(ordered) % 2 == 1
        else (ordered[mid - 1] + ordered[mid]) / 2
    )
    result = {
        "count": len(lengths),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": sum(lengths) / len(lengths),
        "median": float(median),
    }

    # 分位数：决定 max_seq_length 时比 max 更有参考价值
    # （max 常被极少数长尾样本拉高，按它设阈值会浪费大量算力）
    for q in (0.90, 0.95, 0.99):
        idx = min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)
        result[f"p{int(q * 100)}"] = float(ordered[idx])

    return result
