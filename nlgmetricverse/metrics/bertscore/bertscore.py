"""
Bertscore metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.bertscore.bertscore_planet import BertscorePlanet

__main_class__ = "Bertscore"


class Bertscore(MetricAlias):
    """
    Bertscore metric superclass.
    """
    _SUBCLASS = BertscorePlanet
