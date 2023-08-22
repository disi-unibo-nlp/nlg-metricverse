# coding=utf-8

""" Gunning-Fog metric. """

import evaluate
from typing import Callable

import numpy as np
from textstat import textstat

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration

_CITATION = """

"""

_DESCRIPTION = """\
The Gunning fog index is a readability test for English writing in linguistics. The index calculates the number of years of formal
education required to understand the text on the first reading. For example, a fog index of 12 necessitates the reading level of 
a senior in high school in the United States (around 18 years old). Robert Gunning, an American businessman who had previously 
worked in newspaper and textbook publishing, created the test in 1952.
The fog index is frequently used to ensure that text can be easily read by the intended audience. Texts aimed at a broad audience 
typically require a fog index of less than 12. Texts requiring near-universal comprehension typically require an index of less than 8.
"""

_KWARGS_DESCRIPTION = """\
Args:
    predictions: list of predictions to score. Each prediction should be a string.
    references: list of reference to score against. Each reference should be a string.
    reduce_fn: function to reduce score list into a single score. If None, will return a dict of scores.
    n: n-gram order. Default 1.

Returns:
    'score': Gunning-Fog score.

Examples:
    >>> from nlgmetricverse import NLGMetricverse, load_metric
    >>> predictions = ["There is a cat on the mat.", "Look! a wonderful day."]
    >>> references = ["The cat is playing on the mat.", "Today is a wonderful day"]
    >>> scorer = NLGMetricverse(metrics=load_metric("gunning_fog"))
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    {"gunning_fog": {"score": 2.2 }}
"""

_LICENSE = """

"""

CHECKPOINT_URLS = {

}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class GunningFogPlanet(MetricForLanguageGeneration):
    def _info(self):
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=10,
            lower_bound=0,
            features=self._default_features,
            codebase_urls=[""],
            reference_urls=[
                ""
            ],
        )

    def _compute_single_pred_single_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Compute the gunning_fog score for a single prediction and a single reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a single text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            n (int, optional): n-gram order. Default 1.
        """
        result = self.__compute_gunning_fog(predictions)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Compute the gunning_fog score for a single prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            n (int, optional): n-gram order. Default 1.
        """
        result = self.__compute_gunning_fog(predictions)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Compute the gunning_fog score for multiple prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a multiple text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            n (int, optional): n-gram order. Default 1.
        """
        res = []
        for prediction in predictions:
            score = self.__compute_gunning_fog(prediction)
            res.append(score)
        result = np.mean(res)
        return {"score": result}

    @staticmethod
    def __compute_gunning_fog(predictions):
        """
        The purpose of this method is to compute the Gunning Fog index for each prediction in the predictions parameter 
        and return the average score.

            :param predictions: list of predictions to score. Each prediction should be a string.

            :return: average Gunning Fog score
        """
        scores = []
        for prediction in predictions:
            scores.append(textstat.gunning_fog(prediction))
        result = np.mean(scores)
        return result
