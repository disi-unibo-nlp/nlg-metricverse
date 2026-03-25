"""
Narrative Score metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.narrative_score.narrative_score_planet import NarrativeScorePlanet

__main_class__ = "NarrativeScore"


class NarrativeScore(MetricAlias):
    """
    Narrative Score metric superclass.
    Evaluates Chinese autobiographical narrative quality across six dimensions.
    """
    _SUBCLASS = NarrativeScorePlanet
