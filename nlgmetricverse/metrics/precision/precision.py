"""
Precision metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.precision.precision_planet import PrecisionPlanet

__main_class__ = "Precision"


class Precision(MetricAlias):
    """
    Precision metric superclass.
    """
    _SUBCLASS = PrecisionPlanet
