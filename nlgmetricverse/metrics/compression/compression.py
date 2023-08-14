"""
Compression metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.compression.compression_planet import CompressionPlanet

__main_class__ = "Compression"


class Compression(MetricAlias):
    """
    Compression metric superclass.
    """
    _SUBCLASS = CompressionPlanet
