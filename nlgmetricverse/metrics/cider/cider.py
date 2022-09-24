"""
Cider metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.cider.cider_planet import CiderPlanet

__main_class__ = "Cider"


class Cider(MetricAlias):
    """
    Cider metric superclass.
    """
    _SUBCLASS = CiderPlanet
