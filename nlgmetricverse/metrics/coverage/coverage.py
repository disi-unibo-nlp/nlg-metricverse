"""
Coverage metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.coverage.coverage_planet import CoveragePlanet

__main_class__ = "Coverage"


class Coverage(MetricAlias):
    """
    Coverage metric superclass.
    """
    _SUBCLASS = CoveragePlanet
