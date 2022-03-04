from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.meteor.meteor_for_language_generation import MeteorForLanguageGeneration

__main_class__ = "Meteor"


class Meteor(MetricAlias):
    _SUBCLASS = MeteorForLanguageGeneration
