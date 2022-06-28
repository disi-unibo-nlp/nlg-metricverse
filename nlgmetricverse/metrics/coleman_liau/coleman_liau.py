"""
ColemanLiau metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.coleman_liau.coleman_liau_planet import ColemanLiauPlanet

__main_class__ = "ColemanLiau"


class ColemanLiau(MetricAlias):
    """
    ColemanLiau metric superclass..
    """
    _SUBCLASS = ColemanLiauPlanet
