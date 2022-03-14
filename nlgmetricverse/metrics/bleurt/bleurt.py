"""
Bleurt metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.bleurt.bleurt_planet import BleurtPlanet

__main_class__ = "Bleurt"


class Bleurt(MetricAlias):
    """
    Bleurt metric superclass.
    """
    _SUBCLASS = BleurtPlanet
