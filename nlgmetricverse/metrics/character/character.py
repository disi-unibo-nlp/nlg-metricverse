"""
CharacTER metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.character.character_planet import CharacTERPlanet

__main_class__ = "CharacTER"


class CharacTER(MetricAlias):
    """
    CharacTER metric superclass.
    """
    _SUBCLASS = CharacTERPlanet
