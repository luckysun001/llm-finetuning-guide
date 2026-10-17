"""
超参数搜索空间定义

设计目标：
- 不引入 optuna / ray 等重型依赖，只用标准库即可完成网格搜索与随机搜索
- 参数空间既能用于网格搜索（离散枚举），也能用于随机搜索（连续采样）
- 与项目配置解耦：搜索空间只负责"产生参数组合"，如何应用由调用方决定

支持的参数类型：
- ``categorical``  离散枚举（如 batch_size=[4, 8, 16]）
- ``uniform``      区间均匀采样（如 dropout=[0.0, 0.3]）
- ``loguniform``   对数均匀采样（如 learning_rate=[1e-5, 1e-3]）
- ``int``          整数区间（含端点，如 num_train_epochs=[1, 5]）
- ``int_log``      整数对数区间（如 lora.r=[8, 64]）
"""

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

__all__ = ["Parameter", "SearchSpace"]

_KINDS = ("categorical", "uniform", "loguniform", "int", "int_log")


class Parameter:
    """单个超参数的定义

    Args:
        name: 参数名（可用 ``lora.r`` 这类点号路径表示嵌套配置）
        kind: 参数类型，见模块文档
        values: ``categorical`` 的候选值列表
        low: 区间下界（``uniform`` / ``loguniform`` / ``int`` / ``int_log``）
        high: 区间上界
        grid: 网格搜索时使用的离散取值；不指定时由类型自动生成
    """

    def __init__(
        self,
        name: str,
        kind: str = "categorical",
        values: Optional[Sequence[Any]] = None,
        low: Optional[float] = None,
        high: Optional[float] = None,
        grid: Optional[Sequence[Any]] = None,
    ):
        if kind not in _KINDS:
            raise ValueError(f"不支持的参数类型: {kind}，可选: {_KINDS}")
        if kind == "categorical" and not values:
            raise ValueError(f"参数 {name} 为 categorical，必须提供 values")
        if kind != "categorical" and (low is None or high is None):
            raise ValueError(f"参数 {name} 为 {kind}，必须提供 low 与 high")
        if kind in ("loguniform", "int_log") and low is not None and low <= 0:
            raise ValueError(f"参数 {name} 为对数类型，low 必须 > 0，当前为 {low}")

        self.name = name
        self.kind = kind
        self.values = list(values) if values is not None else None
        self.low = low
        self.high = high
        self._grid = list(grid) if grid is not None else None

    # ---------- 网格搜索 ----------

    def grid_values(self, max_points: int = 5) -> List[Any]:
        """返回网格搜索使用的离散取值

        连续类型（uniform / loguniform / int / int_log）默认取 ``max_points`` 个采样点，
        端点必定包含在内，保证搜索覆盖边界。
        """
        if self._grid is not None:
            return list(self._grid)
        if self.kind == "categorical":
            return list(self.values or [])

        low, high = float(self.low), float(self.high)
        if self.kind == "int" or self.kind == "int_log":
            n = min(max_points, int(high) - int(low) + 1)
            if n <= 1:
                return [int(low)]
            if self.kind == "int":
                step = (high - low) / (n - 1)
                raw = [int(round(low + i * step)) for i in range(n)]
            else:
                raw = [
                    int(round(math.exp(
                        math.log(low) + i * (math.log(high) - math.log(low)) / (n - 1)
                    )))
                    for i in range(n)
                ]
            # 去重并保持顺序
            out: List[int] = []
            for v in raw:
                if v not in out:
                    out.append(v)
            return out

        if self.kind == "loguniform":
            return [
                math.exp(math.log(low) + i * (math.log(high) - math.log(low)) / (max_points - 1))
                for i in range(max_points)
            ]
        # uniform
        return [low + i * (high - low) / (max_points - 1) for i in range(max_points)]

    # ---------- 随机搜索 ----------

    def sample(self, rng: random.Random) -> Any:
        """按参数类型随机采样一个取值"""
        if self.kind == "categorical":
            return rng.choice(self.values or [])

        low, high = float(self.low), float(self.high)
        if self.kind == "uniform":
            return rng.uniform(low, high)
        if self.kind == "loguniform":
            return math.exp(rng.uniform(math.log(low), math.log(high)))
        if self.kind == "int":
            return rng.randint(int(low), int(high))
        # int_log
        return int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))

    def __repr__(self) -> str:
        if self.kind == "categorical":
            return f"<Parameter {self.name}={self.kind}{self.values}>"
        return f"<Parameter {self.name}={self.kind}[{self.low}, {self.high}]>"


@dataclass
class SearchSpace:
    """超参数搜索空间（若干 :class:`Parameter` 的集合）

    示例::

        space = SearchSpace([
            Parameter("learning_rate", "loguniform", low=1e-5, high=1e-3),
            Parameter("lora.r", "int_log", low=8, high=64),
            Parameter("per_device_train_batch_size", "categorical", values=[4, 8, 16]),
        ])
    """

    parameters: List[Parameter] = field(default_factory=list)
    seed: int = 42

    def add(self, parameter: Parameter) -> "SearchSpace":
        """追加一个参数，返回自身以支持链式调用"""
        self.parameters.append(parameter)
        return self

    def __len__(self) -> int:
        return len(self.parameters)

    # ---------- 遍历 ----------

    def grid(self, max_points: int = 5) -> Iterator[Dict[str, Any]]:
        """笛卡尔积遍历全部网格组合"""
        names = [p.name for p in self.parameters]
        value_lists = [p.grid_values(max_points) for p in self.parameters]
        for combo in itertools.product(*value_lists):
            yield dict(zip(names, combo))

    def sample(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """随机采样一组参数"""
        rng = rng or random.Random(self.seed)
        return {p.name: p.sample(rng) for p in self.parameters}

    # ---------- 其他 ----------

    def grid_size(self, max_points: int = 5) -> int:
        """网格搜索的总组合数（用于评估搜索开销）"""
        size = 1
        for p in self.parameters:
            size *= len(p.grid_values(max_points))
        return size

    def names(self) -> List[str]:
        return [p.name for p in self.parameters]

    def describe(self) -> str:
        """人类可读的空间描述，便于打印到日志"""
        lines = [f"搜索空间（{len(self.parameters)} 个参数）:"]
        for p in self.parameters:
            if p.kind == "categorical":
                lines.append(f"  - {p.name}: {p.kind} {p.values}")
            else:
                lines.append(f"  - {p.name}: {p.kind} [{p.low}, {p.high}]")
        return "\n".join(lines)
