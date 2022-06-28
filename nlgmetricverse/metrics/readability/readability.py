"""
Readability metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.readability.readability_planet import ReadabilityPlanet

__main_class__ = "Readability"


class Readability(MetricAlias):
    """
    Readability metric superclass..
    """
    _SUBCLASS = ReadabilityPlanet
