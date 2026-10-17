"""
超参数搜索执行器

与具体的训练框架无关：调用方提供一个 ``objective(params) -> float`` 目标函数，
由本模块负责产生参数组合、执行、收集结果并给出最优解。

目标函数可以是真实训练（返回验证集 loss / 困惑度），也可以是任何可量化的代理指标，
因此本模块在 CI 中也能被完整测试。
"""

import csv
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

from llm_finetuning.tuning.search_space import Parameter, SearchSpace

__all__ = [
    "Trial",
    "SearchResult",
    "HyperparameterSearcher",
    "apply_params",
    "default_lora_space",
]

ObjectiveFn = Callable[[Dict[str, Any]], float]


@dataclass
class Trial:
    """单次试验记录"""

    trial_id: int
    params: Dict[str, Any]
    score: Optional[float] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.score is not None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """一次完整搜索的结果集合"""

    strategy: str
    direction: str
    trials: List[Trial] = field(default_factory=list)

    @property
    def successful_trials(self) -> List[Trial]:
        return [t for t in self.trials if t.succeeded]

    @property
    def failed_count(self) -> int:
        return len(self.trials) - len(self.successful_trials)

    def _sorted(self) -> List[Trial]:
        trials = self.successful_trials
        reverse = self.direction == "maximize"
        return sorted(trials, key=lambda t: t.score, reverse=reverse)  # type: ignore[arg-type,return-value]

    @property
    def best_trial(self) -> Optional[Trial]:
        ranked = self._sorted()
        return ranked[0] if ranked else None

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        best = self.best_trial
        return dict(best.params) if best else None

    @property
    def best_score(self) -> Optional[float]:
        best = self.best_trial
        return best.score if best else None

    def top_k(self, k: int = 5) -> List[Trial]:
        """返回得分最好的 k 次试验"""
        return self._sorted()[:k]

    def to_records(self) -> List[Dict[str, Any]]:
        """展平为表格记录，便于导出 CSV"""
        records = []
        for t in self.trials:
            row: Dict[str, Any] = {
                "trial_id": t.trial_id,
                "score": t.score,
                "error": t.error,
                "duration_seconds": round(t.duration_seconds, 4),
            }
            row.update(t.params)
            records.append(row)
        return records

    def save_csv(self, filepath: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
        records = self.to_records()
        if not records:
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                f.write("")
            return filepath
        fieldnames: List[str] = []
        for r in records:
            for key in r:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        return filepath

    def save_json(self, filepath: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
        payload = {
            "strategy": self.strategy,
            "direction": self.direction,
            "total_trials": len(self.trials),
            "failed_trials": self.failed_count,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "trials": [t.to_dict() for t in self.trials],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return filepath

    def summary(self) -> str:
        """人类可读的结果摘要"""
        lines = [
            f"搜索策略: {self.strategy}",
            f"优化方向: {self.direction}",
            f"试验总数: {len(self.trials)}（失败 {self.failed_count} 次）",
        ]
        best = self.best_trial
        if best:
            lines.append(f"最优得分: {best.score}")
            lines.append(f"最优参数: {best.params}")
        else:
            lines.append("没有成功的试验")
        return "\n".join(lines)


class HyperparameterSearcher:
    """超参数搜索器

    Args:
        space: 搜索空间
        objective: 目标函数，接收一组参数并返回标量得分
        direction: ``"minimize"``（默认，如 loss / 困惑度）或 ``"maximize"``（如准确率）
        max_trials: 最多执行的试验次数，超出后提前停止（网格搜索会截断）
        seed: 随机种子
        fail_fast: 目标函数抛异常时是否立即中断，默认记录错误并继续
        on_trial_end: 每次试验结束后的回调，可用于打印进度

    示例::

        searcher = HyperparameterSearcher(space, objective, direction="minimize")
        result = searcher.search(strategy="random", n_trials=20)
        print(result.best_params)
    """

    def __init__(
        self,
        space: SearchSpace,
        objective: ObjectiveFn,
        direction: str = "minimize",
        max_trials: Optional[int] = None,
        seed: int = 42,
        fail_fast: bool = False,
        on_trial_end: Optional[Callable[[Trial], None]] = None,
    ):
        if direction not in ("minimize", "maximize"):
            raise ValueError("direction 只能是 'minimize' 或 'maximize'")
        if len(space) == 0:
            raise ValueError("搜索空间为空，请先添加参数")

        self.space = space
        self.objective = objective
        self.direction = direction
        self.max_trials = max_trials
        self.seed = seed
        self.fail_fast = fail_fast
        self.on_trial_end = on_trial_end

    # ---------- 执行 ----------

    def _run_trial(self, trial_id: int, params: Dict[str, Any]) -> Trial:
        start = time.time()
        try:
            score = float(self.objective(params))
            return Trial(
                trial_id=trial_id,
                params=dict(params),
                score=score,
                duration_seconds=time.time() - start,
            )
        except Exception as exc:  # noqa: BLE001 - 搜索中单次失败不应中断全局
            if self.fail_fast:
                raise
            return Trial(
                trial_id=trial_id,
                params=dict(params),
                score=None,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=time.time() - start,
            )

    # 超过这个组合数就不再完整枚举，避免笛卡尔积爆炸
    _MAX_ENUMERATE = 20000

    def _select_grid_combos(self, max_points: int) -> List[Dict[str, Any]]:
        """挑选本次网格搜索要跑的参数组合

        直接按字典序遍历并截断会有严重的**顺序偏差**：前几组只在第一个参数上有变化，
        其余维度全被锁死在首个取值上。因此这里在需要截断时改为均匀抽样，
        保证每个维度都能被覆盖到。
        """
        estimated = self.space.grid_size(max_points)

        if not self.max_trials or estimated <= self.max_trials:
            return list(self.space.grid(max_points))

        if estimated > self._MAX_ENUMERATE:
            # 空间太大：退化成不重复的随机采样
            rng = random.Random(self.seed)
            combos: List[Dict[str, Any]] = []
            seen: set = set()
            guard = 0
            limit = self.max_trials * 20
            while len(combos) < self.max_trials and guard < limit:
                guard += 1
                sample = self.space.sample(rng)
                key = repr(sorted(sample.items()))
                if key in seen:
                    continue
                seen.add(key)
                combos.append(sample)
            return combos

        combos = list(self.space.grid(max_points))
        total = len(combos)
        n = self.max_trials
        if n == 1:
            return [combos[0]]
        # 等间隔抽样，首尾必取
        indices = {round(i * (total - 1) / (n - 1)) for i in range(n)}
        return [combos[i] for i in sorted(indices)]

    def grid_search(self, max_points: int = 5) -> SearchResult:
        """网格搜索：遍历（或均匀抽样）搜索空间的笛卡尔积"""
        result = SearchResult(strategy="grid", direction=self.direction)
        for i, params in enumerate(self._select_grid_combos(max_points), start=1):
            trial = self._run_trial(i, params)
            result.trials.append(trial)
            if self.on_trial_end:
                self.on_trial_end(trial)
        return result

    def random_search(self, n_trials: int = 10) -> SearchResult:
        """随机搜索：从搜索空间独立采样 n_trials 次"""
        rng = random.Random(self.seed)
        result = SearchResult(strategy="random", direction=self.direction)
        total = n_trials
        if self.max_trials:
            total = min(total, self.max_trials)
        for i in range(1, total + 1):
            params = self.space.sample(rng)
            trial = self._run_trial(i, params)
            result.trials.append(trial)
            if self.on_trial_end:
                self.on_trial_end(trial)
        return result

    def search(self, strategy: str = "grid", **kwargs) -> SearchResult:
        """统一入口，``strategy`` 可选 ``grid`` / ``random``"""
        if strategy == "grid":
            return self.grid_search(**kwargs)
        if strategy == "random":
            return self.random_search(**kwargs)
        raise ValueError(f"未知搜索策略: {strategy}，可选: grid, random")


# ---------- 与项目配置的桥接 ----------


def apply_params(config: Any, params: Dict[str, Any], strict: bool = False) -> List[str]:
    """把一组搜索参数写回 :class:`~llm_finetuning.config.Config`

    支持点号路径，例如 ``lora.r`` 会写入 ``config.lora_config.r``。
    目前识别的顶层前缀：``training``、``data``、``evaluation``、``lora``；
    不带前缀的键会依次在 ``training`` / ``data`` / ``evaluation`` 上查找。

    Args:
        config: 目标配置对象
        params: 参数字典
        strict: 为 True 时，遇到无法写入的键直接抛 ``KeyError``

    Returns:
        成功应用的参数名列表
    """
    prefix_map = {
        "training": "training",
        "data": "data",
        "evaluation": "evaluation",
        "eval": "evaluation",
        "lora": "lora_config",
        "lora_config": "lora_config",
    }

    applied: List[str] = []
    for key, value in params.items():
        parts = key.split(".")
        if parts[0] in prefix_map and len(parts) > 1:
            section_name = prefix_map[parts[0]]
            section = getattr(config, section_name, None)
            attr = parts[1]
        else:
            section = None
            attr = parts[0]

        if section is None:
            # 不带前缀：按 training / data / evaluation 顺序查找
            for candidate in ("training", "data", "evaluation"):
                obj = getattr(config, candidate, None)
                if obj is not None and hasattr(obj, attr):
                    section = obj
                    break

        if section is not None and hasattr(section, attr):
            setattr(section, attr, value)
            applied.append(key)
            continue

        # 回退：直接写入配置对象本身（如 learning_rate 之外的自定义字段）
        if hasattr(config, attr):
            setattr(config, attr, value)
            applied.append(key)
            continue

        if strict:
            raise KeyError(f"配置中不存在可写入的字段: {key}")

    return applied


def default_lora_space() -> SearchSpace:
    """LoRA / QLoRA 微调的常用搜索空间（经验值）

    学习率用对数区间（跨数量级搜索更有效），LoRA 秩用整数对数区间，
    batch size 与 warmup 用离散枚举。
    """
    return SearchSpace(
        [
            Parameter("learning_rate", "loguniform", low=1e-5, high=5e-4),
            Parameter("lora.r", "int_log", low=8, high=64),
            Parameter("lora.lora_alpha", "int_log", low=16, high=128),
            Parameter("training.per_device_train_batch_size", "categorical", values=[2, 4, 8]),
            Parameter("training.warmup_steps", "categorical", values=[0, 50, 100]),
        ]
    )
