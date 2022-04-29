"""
CER metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.cer.cer_planet import CERPlanet

__main_class__ = "CER"


class CER(MetricAlias):
    """
    CER metric superclass..
    """
    _SUBCLASS = CERPlanet
