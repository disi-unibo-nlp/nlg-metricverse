"""
WER metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.wer.wer_planet import WERPlanet

__main_class__ = "WER"


class WER(MetricAlias):
    """
    WER metric super class.
    """
    _SUBCLASS = WERPlanet
