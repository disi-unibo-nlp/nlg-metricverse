# coding=utf-8

""" Flesch-Kincaid metric. """

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
The Flesch-Kincaid readability tests are used to determine how difficult a passage in English is to understand. The Flesch 
Reading-Ease and Flesch-Kincaid Grade Level tests are available. Although they use the same core measures (word length and sentence 
length), the weighting factors are different.
The results of the two tests are roughly inversely related: a text with a relatively high Reading Ease score should have a lower 
Grade-Level score. The Reading Ease evaluation was developed by Rudolf Flesch; later, he and J. Peter Kincaid developed the Grade Level 
evaluation for the United States Navy.
In the Flesch reading-ease test, higher scores indicate material that is easier to read; lower numbers mark passages that are 
more difficult to read.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: list of predictions to score. Each predictions should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each reference should be a string with tokens separated by spaces.
Returns:
    'score': Flesch-Kincaid score.
Examples:
    >>> predictions = ["There is a cat on the mat.", "Look! a wonderful day."]
    >>> references = ["The cat is playing on the mat.", "Today is a wonderful day"]
    >>> scorer = FleschKincaidPlanet()
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    {"flesch_kincaid": { "score": 1.25 }}

"""

_LICENSE = """

"""

CHECKPOINT_URLS = {

}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class FleschKincaidPlanet(MetricForLanguageGeneration):
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
        Compute the flesh_kincaid score for a single prediction and a single reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a single text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
        """
        result = self.__compute_flesch_kincaid(predictions)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Compute the flesh_kincaid score for a single prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
        """
        result = self.__compute_flesch_kincaid(predictions)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Compute the flesh_kincaid score for multiple prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a multiple text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
        """
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
