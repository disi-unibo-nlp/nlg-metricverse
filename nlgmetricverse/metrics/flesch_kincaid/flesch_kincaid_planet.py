# coding=utf-8

""" Flesch-Kincaid metric. """

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
class FleschKincaidPlanet(MetricForLanguageGeneration):
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
        result = self.__compute_flesch_kincaid(predictions)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        result = self.__compute_flesch_kincaid(predictions)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        res = []
        for prediction in predictions:
            score = self.__compute_flesch_kincaid(prediction)
            res.append(score)
        result = np.mean(res)
        return {"score": result}

    @staticmethod
    def __compute_flesch_kincaid(predictions):
        tot_pred = 0
        result = 0
        for prediction in predictions:
            result += textstat.flesch_kincaid_grade(prediction)
            tot_pred = tot_pred + 1
        result = result / tot_pred
        '''total_words = 0
        total_syllables = 0

        total_sentences = len(predictions)
        for sentence in predictions:
            total_words += len(sentence.split())
            total_syllables += syllables.estimate(sentence)

        result = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        result = result / 100'''
        return result
