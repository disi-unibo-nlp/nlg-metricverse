"""
Comet metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.comet.comet_planet import CometPlanet

__main_class__ = "Comet"


class Comet(MetricAlias):
    """
    Comet metric superclass.
    """
    _SUBCLASS = CometPlanet
