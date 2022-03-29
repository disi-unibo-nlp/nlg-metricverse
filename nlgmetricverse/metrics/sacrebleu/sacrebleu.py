from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.sacrebleu.sacrebleu_planet import SacrebleuPlanet

__main_class__ = "Sacrebleu"


class Sacrebleu(MetricAlias):
    _SUBCLASS = SacrebleuPlanet
