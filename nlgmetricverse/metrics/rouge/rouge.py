from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.rouge.rouge_planet import RougePlanet

__main_class__ = "Rouge"


class Rouge(MetricAlias):
    _SUBCLASS = RougePlanet
