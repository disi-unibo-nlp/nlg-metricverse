"""
EED metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.eed.eed_planet import EEDPlanet

__main_class__ = "EED"


class EED(MetricAlias):
    """
    EED metric superclass.
    """
    _SUBCLASS = EEDPlanet
