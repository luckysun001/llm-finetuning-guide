"""数据清洗与去重模块测试"""
import pytest

from llm_finetuning.data.cleaner import (
    CleanReport,
    DataCleaner,
    deduplicate,
    length_stats,
)


SAMPLES = [
    {"text": "第一条样本"},
    {"text": "第二条样本"},
    {"text": "第一条样本"},          # 精确重复
    {"text": "   "},                  # 空白
    {"text": "短"},                   # 过短（min_length=5）
]


class TestClean:
    """基础清洗行为"""

    def test_drops_duplicates_and_empty(self):
        cleaned, report = DataCleaner(min_length=1).clean(SAMPLES[:4])

        assert len(cleaned) == 2
        assert report.total == 4
        assert report.removed_duplicate == 1
        assert report.removed_empty == 1

    def test_min_length_filter(self):
        cleaned, report = DataCleaner(min_length=5).clean(SAMPLES)

        assert all(len(r["text"]) >= 5 for r in cleaned)
        assert report.removed_short == 1  # "短"

    def test_max_length_filter(self):
        records = [{"text": "x" * 10}, {"text": "y" * 100}]
        cleaned, report = DataCleaner(max_length=50).clean(records)

        assert len(cleaned) == 1
        assert report.removed_long == 1

    def test_missing_field_is_removed(self):
        cleaned, report = DataCleaner().clean([{"text": "ok"}, {"other": "no text"}])

        assert len(cleaned) == 1
        assert report.removed_missing_field == 1

    def test_keeps_other_fields(self):
        records = [{"text": " 问题 ", "instruction": "问", "output": "答"}]
        cleaned, _ = DataCleaner().clean(records)

        assert cleaned[0]["instruction"] == "问"
        assert cleaned[0]["output"] == "答"
        assert cleaned[0]["text"] == "问题"  # 已归一化

    def test_deduplicate_can_be_disabled(self):
        records = [{"text": "same"}, {"text": "same"}]
        cleaned, report = DataCleaner(deduplicate=False).clean(records)

        assert len(cleaned) == 2
        assert report.removed_duplicate == 0

    def test_empty_input(self):
        cleaned, report = DataCleaner().clean([])
        assert cleaned == []
        assert report.total == 0
        assert report.kept == 0


class TestNormalization:
    """文本归一化"""

    def test_collapses_whitespace(self):
        cleaned, _ = DataCleaner().clean([{"text": "a\n\n  b\t　c"}])
        assert cleaned[0]["text"] == "a b c"

    def test_fullwidth_to_halfwidth(self):
        cleaned, _ = DataCleaner().clean([{"text": "ＡＢＣ１２３"}])
        assert cleaned[0]["text"] == "ABC123"

    def test_lowercase_option(self):
        cleaned, _ = DataCleaner(lowercase=True).clean([{"text": "Hello World"}])
        assert cleaned[0]["text"] == "hello world"

    def test_normalize_can_be_disabled(self):
        cleaned, _ = DataCleaner(normalize_whitespace=False).clean([{"text": "a  b"}])
        assert cleaned[0]["text"] == "a  b"


class TestFuzzyDedup:
    """模糊去重：忽略标点与空白差异"""

    def test_punctuation_only_difference_is_deduped(self):
        records = [{"text": "你好，世界！"}, {"text": "你好世界"}]
        cleaned, report = DataCleaner(fuzzy_dedup=True).clean(records)

        assert len(cleaned) == 1
        assert report.removed_duplicate == 1

    def test_exact_mode_keeps_them(self):
        records = [{"text": "你好，世界！"}, {"text": "你好世界"}]
        cleaned, _ = DataCleaner(fuzzy_dedup=False).clean(records)
        assert len(cleaned) == 2


class TestReport:
    """清洗报告"""

    def test_removed_property(self):
        _, report = DataCleaner(min_length=1).clean(SAMPLES[:4])
        assert report.removed == report.total - report.kept

    def test_summary_contains_counts(self):
        _, report = DataCleaner(min_length=1).clean(SAMPLES[:4])
        summary = report.summary()
        assert "总计 4 条" in summary
        assert "保留 2 条" in summary

    def test_to_dict_keys(self):
        data = CleanReport(total=10, kept=7).to_dict()
        for key in ("total", "kept", "removed", "removed_duplicate"):
            assert key in data

    def test_keep_removed_samples(self):
        _, report = DataCleaner(keep_report_samples=2).clean(SAMPLES)
        assert len(report.removed_samples) <= 2
        assert all("_removed_reason" in s for s in report.removed_samples)


class TestHelpers:
    """便捷函数"""

    def test_deduplicate(self):
        assert len(deduplicate([{"text": "a"}, {"text": "a"}, {"text": "b"}])) == 2

    def test_length_stats(self):
        stats = length_stats([{"text": "ab"}, {"text": "abcd"}, {"text": "abc"}])
        assert stats["count"] == 3
        assert stats["min"] == 2.0
        assert stats["max"] == 4.0
        assert stats["median"] == 3.0
        assert stats["mean"] == pytest.approx(3.0)

    def test_length_stats_empty(self):
        assert length_stats([])["count"] == 0


class TestValidation:
    """参数校验"""

    def test_negative_min_length(self):
        with pytest.raises(ValueError):
            DataCleaner(min_length=-1)

    def test_max_smaller_than_min(self):
        with pytest.raises(ValueError):
            DataCleaner(min_length=10, max_length=5)
