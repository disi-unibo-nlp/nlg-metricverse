from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.rouge.rouge_for_language_generation import RougeForLanguageGeneration

__main_class__ = "Rouge"


class Rouge(MetricAlias):
    _SUBCLASS = RougeForLanguageGeneration
