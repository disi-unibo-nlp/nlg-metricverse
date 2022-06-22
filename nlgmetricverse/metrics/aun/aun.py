"""
AUN metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.aun.aun_planet import AUNPlanet

__main_class__ = "AUN"


class AUN(MetricAlias):
    """
    AUN metric super class.
    """
    _SUBCLASS = AUNPlanet
