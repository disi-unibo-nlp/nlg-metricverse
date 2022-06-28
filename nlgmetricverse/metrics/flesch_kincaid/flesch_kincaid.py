"""
FleschKincaid metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.flesch_kincaid.flesch_kincaid_planet import FleschKincaidPlanet

__main_class__ = "FleschKincaid"


class FleschKincaid(MetricAlias):
    """
    FleschKincaid metric superclass..
    """
    _SUBCLASS = FleschKincaidPlanet
