"""
F1 metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.f1.f1_planet import F1Planet

__main_class__ = "F1"


class F1(MetricAlias):
    """
    F1 metric superclass.
    """
    _SUBCLASS = F1Planet
