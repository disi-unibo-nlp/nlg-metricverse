# coding=utf-8

""" Coleman-Liau metric. """

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
The Coleman-Liau index is a readability test developed by Meri Coleman and T. L. Liau to assess text comprehension. Its output, 
like the Flesch-Kincaid Grade Level and Gunning-Fog index, approximates the grade level thought necessary to comprehend the text 
in the United States. 
Coleman-Liau, like the ARI, but unlike most of the other indices, is based on characters rather than syllables per word. Although 
opinions differ on its accuracy in comparison to the syllable/word and complex word indices, computer programs count characters more 
easily and accurately than syllables.
The Coleman-Liau index was created to be easily calculated mechanically from hard-copy text samples. It does not require the character
content of words to be analyzed, unlike syllable-based readability indices, only their length in characters. As a result, it could be 
used with theoretically simple mechanical scanners that only need to recognize character, word, and sentence boundaries, eliminating 
the need for full optical character recognition or manual keypunching.
"""

_KWARGS_DESCRIPTION = """\
Args:
    predictions (EvaluationInstance): A list of strings containing the predicted sentences.
    references (EvaluationInstance): A list of strings containing the reference sentences.
Returns:
    score (`float`): The Coleman-Liau index.
"""

_LICENSE = """

"""

CHECKPOINT_URLS = {

}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ColemanLiauPlanet(MetricForLanguageGeneration):
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
        Compute the coleman_liau score for a single prediction and a single reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a single text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores. 
            n (int, optional): Number of samples to evaluate.
        """
        result = self.__compute_coleman_liau(predictions)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Compute the coleman_liau score for a single prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            n (int, optional): Number of samples to evaluate.
        """
        result = self.__compute_coleman_liau(predictions)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Compute the coleman_liau score for multiple prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a multiple text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            n (int, optional): Number of samples to evaluate.
        """
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
