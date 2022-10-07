"""
Bleu metric super class.
"""
from typing import Any, Dict, Optional
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.bleu.bleu_planet import BleuPlanet

__main_class__ = "Bleu"

from nlgmetricverse.utils.string import camel_to_snake


class Bleu(MetricAlias):
    """
    Bleu metric superclass.
    """
    _SUBCLASS = BleuPlanet

    @classmethod
    def construct(
        cls,
        task: str = None,
        resulting_name: Optional[str] = None,
        compute_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Create a subclass implementing Bleu metric.

        :param task: Metric task.
        :param resulting_name: Metric name.
        :param compute_kwargs: Additional arguments for correct output.
        :param kwargs: Additional arguments used for the metric computation.
        :return: subclass containing the implementation of the metric.
        """
        subclass = cls._get_subclass()
        resulting_name = resulting_name or cls._get_path(compute_kwargs=compute_kwargs)
        return subclass._construct(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    @classmethod
    def _get_path(cls, compute_kwargs: Dict[str, Any] = None) -> str:
        """
        All metric modules must implement this method as it is used to form MetricOutput properly.

        :param compute_kwargs: Additional arguments for correct output.
        :return: Metric name.
        """
        path = camel_to_snake(cls.__name__)
        if compute_kwargs is None:
            return path

        max_order = compute_kwargs.get("max_order")
        if max_order is not None:
            return f"{path}_{max_order}"
