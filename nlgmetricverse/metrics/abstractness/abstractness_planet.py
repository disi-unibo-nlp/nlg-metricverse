# coding=utf-8

""" Abstractness metric. """

import evaluate
from typing import Callable
from nltk import ngrams

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
class AbstractnessPlanet(MetricForLanguageGeneration):
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
        result = self.__compute_abstractness(references, predictions, n)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        result = self.__compute_abstractness(references, predictions, n)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        predList = []
        refList = []
        for pred in predictions:
            predList += pred
        for ref in references:
            refList += ref
        result = self.__compute_abstractness(refList, predList, n)
        return {"score": result}

    @staticmethod
    def __compute_abstractness(res_references, res_predictions, n):
        total_match = 0
        n_words = 0
        for reference, candidate in zip(res_references, res_predictions):
            match = 0
            monograms = candidate.split(" ")
            n_words = n_words + len(monograms)  # count all words in test set
            if n > len(monograms):
                return "Not possible to create " + str(n) + "-grams, too many few words"
            for w2 in ngrams(monograms, n):
                substr = " ".join(w2)
                if substr not in reference:
                    match = match + 1
            # n_words=n_words+1 #counter for total n-gram number
            total_match = total_match + match
        return total_match / n_words
