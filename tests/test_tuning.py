"""
超参数搜索模块测试

目标函数使用简单的解析函数（如 Rosenbrock 变体），
保证测试不依赖 torch / GPU，能在 CI 中秒级完成。
"""

import json
import math
import random

import pytest

from llm_finetuning.config import Config, FinetuningMethod
from llm_finetuning.tuning import (
    HyperparameterSearcher,
    Parameter,
    SearchResult,
    SearchSpace,
    Trial,
    apply_params,
    default_lora_space,
)


# ---------------- Parameter ----------------


def test_categorical_parameter_grid_and_sample():
    p = Parameter("batch_size", "categorical", values=[4, 8, 16])
    assert p.grid_values() == [4, 8, 16]
    rng = random.Random(0)
    assert p.sample(rng) in (4, 8, 16)


def test_uniform_parameter_endpoints_included():
    p = Parameter("dropout", "uniform", low=0.0, high=0.3)
    values = p.grid_values(max_points=4)
    assert values[0] == 0.0
    assert values[-1] == 0.3
    assert len(values) == 4


def test_loguniform_parameter_is_log_spaced():
    p = Parameter("learning_rate", "loguniform", low=1e-5, high=1e-3)
    values = p.grid_values(max_points=3)
    assert values[0] == pytest.approx(1e-5)
    assert values[-1] == pytest.approx(1e-3)
    # 对数间隔意味着几何中点，而非算术中点
    assert values[1] == pytest.approx(1e-4)


def test_int_parameter_grid_is_unique_and_in_range():
    p = Parameter("num_epochs", "int", low=1, high=5)
    values = p.grid_values(max_points=5)
    assert values == [1, 2, 3, 4, 5]


def test_int_log_parameter_stays_within_bounds():
    p = Parameter("lora.r", "int_log", low=8, high=64)
    rng = random.Random(7)
    for _ in range(50):
        value = p.sample(rng)
        assert isinstance(value, int)
        assert 8 <= value <= 64


def test_int_parameter_with_tight_range_collapses_to_single_value():
    p = Parameter("k", "int", low=3, high=3)
    assert p.grid_values() == [3]


def test_parameter_rejects_invalid_kind():
    with pytest.raises(ValueError, match="不支持的参数类型"):
        Parameter("x", "magic")


def test_categorical_parameter_requires_values():
    with pytest.raises(ValueError, match="必须提供 values"):
        Parameter("x", "categorical")


def test_log_parameter_rejects_non_positive_low():
    with pytest.raises(ValueError, match="low 必须 > 0"):
        Parameter("lr", "loguniform", low=0.0, high=1.0)


def test_uniform_parameter_requires_bounds():
    with pytest.raises(ValueError, match="必须提供 low 与 high"):
        Parameter("x", "uniform", low=0.0)


# ---------------- SearchSpace ----------------


def _two_dim_space() -> SearchSpace:
    return SearchSpace(
        [
            Parameter("a", "categorical", values=[1, 2, 3]),
            Parameter("b", "categorical", values=[10, 20]),
        ]
    )


def test_search_space_grid_is_cartesian_product():
    space = _two_dim_space()
    combos = list(space.grid())
    assert len(combos) == 6
    assert {"a": 1, "b": 10} in combos
    assert {"a": 3, "b": 20} in combos


def test_search_space_grid_size_matches():
    space = _two_dim_space()
    assert space.grid_size() == 6


def test_search_space_sample_respects_seed():
    space = _two_dim_space()
    s1 = space.sample(random.Random(42))
    s2 = space.sample(random.Random(42))
    assert s1 == s2


def test_search_space_add_is_chainable():
    space = SearchSpace()
    space.add(Parameter("x", "categorical", values=[1]))
    assert len(space) == 1


def test_search_space_describe_mentions_parameters():
    space = _two_dim_space()
    text = space.describe()
    assert "a" in text and "b" in text


# ---------------- Searcher ----------------


def _quadratic_minimum(params):
    """最小值在 x=3 处"""
    x = params["x"]
    return (x - 3) ** 2


def test_grid_search_finds_minimum():
    space = SearchSpace([Parameter("x", "int", low=0, high=6)])
    searcher = HyperparameterSearcher(space, _quadratic_minimum, direction="minimize")
    result = searcher.grid_search()
    assert result.best_params == {"x": 3}
    assert result.best_score == 0.0


def test_maximize_direction_picks_largest_score():
    space = SearchSpace([Parameter("x", "int", low=0, high=5)])
    searcher = HyperparameterSearcher(
        space, lambda p: -((p["x"] - 5) ** 2), direction="maximize"
    )
    result = searcher.grid_search()
    assert result.best_params == {"x": 5}


def test_random_search_respects_trial_count_and_seed():
    space = SearchSpace([Parameter("x", "int", low=0, high=100)])
    searcher = HyperparameterSearcher(space, _quadratic_minimum, seed=11)
    result = searcher.random_search(n_trials=15)
    assert len(result.trials) == 15

    again = HyperparameterSearcher(space, _quadratic_minimum, seed=11)
    assert again.random_search(n_trials=15).best_params == result.best_params


def test_max_trials_truncates_grid_search():
    space = _two_dim_space()
    searcher = HyperparameterSearcher(space, lambda p: 1.0, max_trials=2)
    result = searcher.grid_search()
    assert len(result.trials) == 2


def test_truncated_grid_search_covers_every_dimension():
    """回归测试：按字典序截断会把后面的维度锁死在首个取值上

    早期实现直接 itertools.product 后 break，导致前 N 组里第一个参数几乎不变、
    其余维度全部固定。这里要求截断后每个维度都有多个不同取值。
    """
    space = SearchSpace(
        [
            Parameter("a", "int", low=0, high=9),
            Parameter("b", "int", low=0, high=9),
        ]
    )
    searcher = HyperparameterSearcher(space, lambda p: 0.0, max_trials=10)
    result = searcher.grid_search()

    assert len(result.trials) == 10
    assert len({t.params["a"] for t in result.trials}) > 1
    assert len({t.params["b"] for t in result.trials}) > 1


def test_grid_search_runs_all_combos_when_budget_is_larger():
    space = _two_dim_space()  # 6 组
    searcher = HyperparameterSearcher(space, lambda p: 0.0, max_trials=100)
    assert len(searcher.grid_search().trials) == 6


def test_huge_space_falls_back_to_sampling_without_explosion():
    """组合数超过枚举上限时，退化为不重复采样而不是生成上千万条组合"""
    space = SearchSpace([Parameter(f"p{i}", "int", low=0, high=9) for i in range(8)])
    searcher = HyperparameterSearcher(space, lambda p: 0.0, max_trials=5)
    result = searcher.grid_search()
    assert len(result.trials) == 5
    # 采样不重复
    keys = {repr(sorted(t.params.items())) for t in result.trials}
    assert len(keys) == 5


def test_failed_trial_is_recorded_not_raised():
    def boom(params):
        raise RuntimeError("显存不足")

    space = SearchSpace([Parameter("x", "categorical", values=[1, 2])])
    searcher = HyperparameterSearcher(space, boom)
    result = searcher.grid_search()
    assert result.failed_count == 2
    assert result.best_trial is None
    assert "显存不足" in result.trials[0].error


def test_fail_fast_reraises():
    def boom(params):
        raise RuntimeError("boom")

    space = SearchSpace([Parameter("x", "categorical", values=[1])])
    searcher = HyperparameterSearcher(space, boom, fail_fast=True)
    with pytest.raises(RuntimeError, match="boom"):
        searcher.grid_search()


def test_search_result_top_k_ordering():
    space = SearchSpace([Parameter("x", "int", low=0, high=6)])
    searcher = HyperparameterSearcher(space, _quadratic_minimum, direction="minimize")
    result = searcher.grid_search()
    top3 = result.top_k(3)
    assert [t.params["x"] for t in top3] == [3, 2, 4]


def test_search_result_summary_contains_best():
    space = SearchSpace([Parameter("x", "int", low=0, high=6)])
    searcher = HyperparameterSearcher(space, _quadratic_minimum)
    summary = searcher.grid_search().summary()
    assert "最优参数" in summary


def test_search_result_export_json_and_csv(tmp_path):
    space = SearchSpace([Parameter("x", "int", low=0, high=4)])
    searcher = HyperparameterSearcher(space, _quadratic_minimum)
    result = searcher.grid_search()

    json_path = result.save_json(str(tmp_path / "search.json"))
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["best_params"] == {"x": 3}
    assert len(payload["trials"]) == 5

    csv_path = result.save_csv(str(tmp_path / "search.csv"))
    with open(csv_path, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    # 1 行表头 + 5 行试验
    assert len(lines) == 6


def test_searcher_unknown_strategy_raises():
    space = SearchSpace([Parameter("x", "categorical", values=[1])])
    searcher = HyperparameterSearcher(space, lambda p: 0.0)
    with pytest.raises(ValueError, match="未知搜索策略"):
        searcher.search(strategy="bayesian")


def test_searcher_validates_direction_and_empty_space():
    space = SearchSpace([Parameter("x", "categorical", values=[1])])
    with pytest.raises(ValueError, match="direction 只能是"):
        HyperparameterSearcher(space, lambda p: 0.0, direction="sideways")
    with pytest.raises(ValueError, match="搜索空间为空"):
        HyperparameterSearcher(SearchSpace(), lambda p: 0.0)


def test_on_trial_end_callback_invoked():
    seen = []

    def objective(params):
        return params["x"]

    space = SearchSpace([Parameter("x", "categorical", values=[1, 2])])
    searcher = HyperparameterSearcher(
        space, objective, direction="maximize", on_trial_end=seen.append
    )
    searcher.grid_search()
    assert len(seen) == 2
    assert all(isinstance(t, Trial) for t in seen)


def test_trial_succeeded_property():
    assert Trial(trial_id=1, params={}, score=1.0).succeeded is True
    assert Trial(trial_id=2, params={}, error="boom").succeeded is False


# ---------------- 与配置的桥接 ----------------


def test_apply_params_with_prefixed_dotted_path():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.LORA)
    applied = apply_params(config, {"lora.r": 32, "training.learning_rate": 1e-4})
    assert applied == ["lora.r", "training.learning_rate"]
    assert config.lora_config.r == 32
    assert config.training.learning_rate == 1e-4


def test_apply_params_without_prefix_finds_section():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.LORA)
    apply_params(config, {"num_train_epochs": 7})
    assert config.training.num_train_epochs == 7


def test_apply_params_skips_unknown_keys_when_not_strict():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.LORA)
    applied = apply_params(config, {"definitely_not_a_field": 1})
    assert applied == []


def test_apply_params_strict_raises_on_unknown_key():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.LORA)
    with pytest.raises(KeyError):
        apply_params(config, {"definitely_not_a_field": 1}, strict=True)


def test_apply_params_supports_adalora_rank_budget():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.ADALORA)
    apply_params(config, {"lora.target_r": 4, "lora.init_r": 16})
    assert config.lora_config.target_r == 4
    assert config.lora_config.init_r == 16


def test_default_lora_space_is_usable():
    space = default_lora_space()
    assert len(space) == 5
    assert space.grid_size() > 0
    sample = space.sample(random.Random(3))
    assert 1e-5 <= sample["learning_rate"] <= 5e-4
    assert 8 <= sample["lora.r"] <= 64


def test_search_loop_with_default_space_and_config_roundtrip():
    """端到端：用默认空间搜索，把最优参数写回 Config"""

    def objective(params):
        # 构造一个"学习率越接近 2e-4 越好"的代理指标
        return abs(math.log(params["learning_rate"] / 2e-4))

    space = default_lora_space()
    searcher = HyperparameterSearcher(space, objective, direction="minimize", seed=5)
    result = searcher.random_search(n_trials=30)

    config = Config("dummy-model", finetuning_method=FinetuningMethod.LORA)
    applied = apply_params(config, result.best_params)
    assert "learning_rate" in applied
    assert config.training.learning_rate == result.best_params["learning_rate"]


# ---------------------------------------------------------------------------
# SearchSpace.from_dict：字典简写构造
# ---------------------------------------------------------------------------


def test_from_dict_parses_tuple_shorthands():
    space = SearchSpace.from_dict(
        {
            "learning_rate": ("loguniform", 1e-5, 1e-3),
            "lora.r": ("int_log", 4, 64),
            "batch_size": ("categorical", [4, 8, 16]),
            "warmup_ratio": ("uniform", 0.0, 0.2),
        }
    )
    kinds = {p.name: p.kind for p in space.parameters}
    assert kinds == {
        "learning_rate": "loguniform",
        "lora.r": "int_log",
        "batch_size": "categorical",
        "warmup_ratio": "uniform",
    }
    assert space.names() == ["learning_rate", "lora.r", "batch_size", "warmup_ratio"]


def test_from_dict_accepts_dict_and_plain_list():
    space = SearchSpace.from_dict(
        {
            "a": {"kind": "uniform", "low": 0.0, "high": 1.0},
            "b": [1, 2, 3],
            "c": ("categorical", 5, 6),
        }
    )
    by_name = {p.name: p for p in space.parameters}
    assert by_name["a"].kind == "uniform"
    assert (by_name["a"].low, by_name["a"].high) == (0.0, 1.0)
    assert by_name["b"].kind == "categorical" and by_name["b"].values == [1, 2, 3]
    assert by_name["c"].values == [5, 6]


def test_from_dict_samples_respect_bounds():
    space = SearchSpace.from_dict(
        {"lr": ("loguniform", 1e-5, 1e-3), "r": ("int_log", 4, 64)}
    )
    rng = random.Random(0)
    for _ in range(50):
        sample = space.sample(rng)
        assert 1e-5 <= sample["lr"] <= 1e-3
        assert 4 <= sample["r"] <= 64


def test_from_dict_rejects_unknown_kind():
    with pytest.raises(ValueError, match="类型不合法"):
        SearchSpace.from_dict({"x": ("not_a_kind", 0.0, 1.0)})


def test_from_dict_rejects_bad_bounds():
    with pytest.raises(ValueError, match=r"\(low, high\)"):
        SearchSpace.from_dict({"x": ("uniform", 0.0)})
