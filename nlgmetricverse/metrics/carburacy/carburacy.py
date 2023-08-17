"""
Carburacy metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.carburacy.carburacy_planet import CarburacyPlanet

__main_class__ = "Carburacy"


class Carburacy(MetricAlias):
    """
    Carburacy metric super class.
    """
    _SUBCLASS = CarburacyPlanet
