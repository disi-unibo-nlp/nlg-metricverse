# coding=utf-8

""" Carburacy metric. """

import evaluate
import math
import numpy as np
from typing import Callable
from collections import namedtuple as _namedtuple

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.metrics._core import MetricForLanguageGeneration


_CITATION = """

"""

_DESCRIPTION = """
"""

_KWARGS_DESCRIPTION = """
Args:
    score: The R value of the prediction.
    co2_val: The CO2 value of the prediction.
Returns:
    carburacy: The average carburacy of the predictions.
Examples:
    >>> scorer = NLGMetricverse(metrics=load_metric("carburacy"))
    >>> predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
    >>> references = ["Peace at home, peace in th world.", "The cat is playing on the mat."]
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    { "total_items": 2, "empty_items": 0, "carburacy": { "score": 0.95 }}
"""

_LICENSE = """

"""

CHECKPOINT_URLS = {

}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CarburacyPlanet(MetricForLanguageGeneration):
    def _info(self):
        """
        Returns metadata about the metric.

        Returns:
            MetricInfo: An object containing metadata about the metric.
        """
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=[""],
            reference_urls=[
                ""
            ],
        )

    def _prepare_carburacy(
            self, reduce_fn: Callable = None, **kwargs
    ):
        """
        Computes the carburacy score for a single predicted text and a single reference text.

        Args:
            reduce_fn (Callable): A function to use for reducing the carburacy scores across multiple examples.
        """
        score = kwargs.get("score", None)
        co2 = kwargs.get("co2_val", None)
        if co2 is not np.nan:
            result = self._compute_carburacy(score, None, co2)
        return {"score": round(result * 100, 2)}

    @staticmethod
    def _compute_carburacy(score, emission_train, emission_test, alpha=10, beta_train=1, beta_test=100):
        carburacy_train = None
        if emission_train is not None:
            carburacy_train = math.exp(math.log(score/100, alpha)) / (1 + emission_train * beta_train)
        carburacy_test = None
        if emission_test is not None:
            carburacy_test = math.exp(math.log(score/100, alpha)) / (1 + emission_test * beta_test)
        carburacy = None
        if carburacy_train is not None and carburacy_test is not None:
            carburacy = (2 * carburacy_train * carburacy_test) / (carburacy_train + carburacy_test)
        return carburacy