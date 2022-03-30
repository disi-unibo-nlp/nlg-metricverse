"""
CHRF metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.chrf.chrf_planet import CHRFPlanet

__main_class__ = "CHRF"


class CHRF(MetricAlias):
    """
    CHRF metric superclass.
    """
    _SUBCLASS = CHRFPlanet
