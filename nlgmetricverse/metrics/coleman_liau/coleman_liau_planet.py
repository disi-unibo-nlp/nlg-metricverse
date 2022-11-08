# coding=utf-8

""" Coleman-Liau metric. """

import evaluate
from typing import Callable

import numpy as np
from textstat import textstat

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.metrics._core import MetricForLanguageGeneration

_CITATION = """

"""

_DESCRIPTION = """

"""

_KWARGS_DESCRIPTION = """

"""

_LICENSE = """

"""

CHECKPOINT_URLS = {

}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ColemanLiauPlanet(MetricForLanguageGeneration):
    def _info(self):
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

    def _compute_single_pred_single_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        result = self.__compute_coleman_liau(predictions)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        result = self.__compute_coleman_liau(predictions)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        res = []
        for prediction in predictions:
            score = self.__compute_coleman_liau(prediction)
            res.append(score)
        result = np.mean(res)
        return {"score": result}

    @staticmethod
    def __compute_coleman_liau(predictions):
        scores = []
        for prediction in predictions:
            scores.append(textstat.coleman_liau_index(prediction))
        result = np.mean(scores)
        return result
