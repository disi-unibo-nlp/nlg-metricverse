"""
GunningFog metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.gunning_fog.gunning_fog_planet import GunningFogPlanet

__main_class__ = "GunningFog"


class GunningFog(MetricAlias):
    """
    GunningFog metric superclass..
    """
    _SUBCLASS = GunningFogPlanet
