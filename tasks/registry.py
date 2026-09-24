"""
tasks/registry.py
Registry cho tất cả benchmarks.

Thêm benchmark mới:
  1. Tạo folder tasks/<tên>/
  2. Implement BenchmarkConfig trong config.py
  3. Thêm import ở cuối file này
  4. Gọi get_benchmark("<tên>") là xong
"""

from typing import Dict, Type
from .base import BenchmarkConfig

_REGISTRY: Dict[str, Type[BenchmarkConfig]] = {}


def register(name: str):
    """Decorator để đăng ký benchmark vào registry."""
    def decorator(cls: Type[BenchmarkConfig]):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_benchmark(task: str) -> BenchmarkConfig:
    """Trả về instance của BenchmarkConfig tương ứng."""
    if task not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown benchmark '{task}'. "
            f"Available: {available}. "
            f"Check tasks/registry.py để thêm benchmark mới."
        )
    return _REGISTRY[task]()


def list_benchmarks():
    return list(_REGISTRY.keys())

from .gsm8k.config import GSM8KConfig                      # noqa: F401, E402
from .strategyqa.config import StrategyQAConfig            # noqa: F401, E402
from .mmlu.config import MMLUConfig                        # noqa: F401, E402
from .commonsenseqa.config import CommonsenseQAConfig      # noqa: F401, E402
from .truthfulqa.config import TruthfulQAConfig            # noqa: F401, E402
from .bbh.config import BBHConfig                          # noqa: F401, E402
