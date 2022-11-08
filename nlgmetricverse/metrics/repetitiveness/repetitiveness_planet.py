# coding=utf-8

""" Repetitiveness metric. """

import evaluate
from typing import Callable

from collections import Counter

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
class RepetitivenessPlanet(MetricForLanguageGeneration):
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

    def _compute_single_pred_single_ref(self, predictions: EvaluationInstance, references: EvaluationInstance,
                                        reduce_fn: Callable = None, **kwargs):
        result = self.__compute_repetitiveness(predictions)
        return {"score": result}

    def _compute_single_pred_multi_ref(self, predictions: EvaluationInstance, references: EvaluationInstance,
                                       reduce_fn: Callable = None, **kwargs):
        result = self.__compute_repetitiveness(predictions)
        return {"score": result}

    def _compute_multi_pred_multi_ref(self, predictions: EvaluationInstance, references: EvaluationInstance,
                                      reduce_fn: Callable = None, **kwargs):
        inputList = []
        for prediction in predictions:
            inputList += prediction
        result = self.__compute_repetitiveness(inputList)
        return {"score": result}

    @staticmethod
    def __compute_repetitiveness(predictions):
        counter = 0
        for candidate in predictions:
            monograms = candidate.split(" ")
            n_words = len(monograms)
            m_counted = Counter(monograms)
            for ngram in m_counted.values():
                if ngram > 1:
                    counter = counter + 1  # if a word  that repeats itself is found
            counter = counter + n_words
        return (counter / len(predictions)) / 10
