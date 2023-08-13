"""
Density metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.density.density_planet import DensityPlanet

__main_class__ = "Density"


class Density(MetricAlias):
    """
    DensityPlanet metric superclass.
    """
    _SUBCLASS = DensityPlanet
