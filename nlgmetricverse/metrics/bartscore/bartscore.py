from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.bartscore.bartscore_for_language_generation import BartscorePlanet

__main_class__ = "Bartscore"


class Bartscore(MetricAlias):
    _SUBCLASS = BartscorePlanet
