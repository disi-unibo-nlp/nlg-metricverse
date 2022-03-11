"""
Bartscore metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.bartscore.bartscore_planet import BartscorePlanet

__main_class__ = "Bartscore"


class Bartscore(MetricAlias):
    """
    Bartscore metric superclass..
    """
    _SUBCLASS = BartscorePlanet
