# coding=utf-8

""" Average Unique N-Gram metric. """

import evaluate
from typing import Callable
from nltk import ngrams

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.metrics._core import MetricForLanguageGeneration
from nlgmetricverse.utils.data_structure import remove_duplicates

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
class AUNPlanet(MetricForLanguageGeneration):
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
        result = self.__compute_average_unique_ngram(predictions, n)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        result = self.__compute_average_unique_ngram(predictions, n)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        inputList = []
        for prediction in predictions:
            inputList += prediction
        result = self.__compute_average_unique_ngram(inputList, n)
        return {"score": result}

    @staticmethod
    def __compute_average_unique_ngram(predictions, n):
        n_grams_count = 0
        unique_n_grams_count = 0

        for candidate in predictions:
            n_grams = list(ngrams(candidate.split(), n))
            for _ in n_grams:
                n_grams_count += 1
            unique_n_grams = remove_duplicates(n_grams)
            unique_n_grams_count += len(unique_n_grams)
        return unique_n_grams_count / n_grams_count

