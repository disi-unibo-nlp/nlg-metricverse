import evaluate
from typing import List
from dataclasses import dataclass

@dataclass
class metric_info(evaluate.MetricInfo):
    """Information about my custom metric."""

    upper_bound: int
    lower_bound: int
