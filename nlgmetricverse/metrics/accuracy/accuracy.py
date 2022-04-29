"""
Accuracy metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.accuracy.accuracy_planet import AccuracyPlanet

__main_class__ = "Accuracy"


class Accuracy(MetricAlias):
    """
    Accuracy metric superclass..
    """
    _SUBCLASS = AccuracyPlanet
