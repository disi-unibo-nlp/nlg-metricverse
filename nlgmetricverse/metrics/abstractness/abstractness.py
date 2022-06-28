"""
Abstractness metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.abstractness.abstractness_planet import AbstractnessPlanet

__main_class__ = "Abstractness"


class Abstractness(MetricAlias):
    """
    Abstractness metric superclass..
    """
    _SUBCLASS = AbstractnessPlanet
