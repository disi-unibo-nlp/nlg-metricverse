from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.bertscore.bertscore_planet import BertscorePlanet

__main_class__ = "Bertscore"


class Bertscore(MetricAlias):
    _SUBCLASS = BertscorePlanet
