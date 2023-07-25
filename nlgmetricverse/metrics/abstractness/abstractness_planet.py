# coding=utf-8

""" Abstractness metric. """

import evaluate
from typing import Callable
from nltk import ngrams

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.metrics._core import MetricForLanguageGeneration

_CITATION = """

"""

_DESCRIPTION = """ A metric for measuring the abstractness of generated text. This metric computes the proportion of n-grams in the generated text that do not appear in the reference text. The abstractness score ranges from 0 to 1, with higher values indicating more abstract text.

    Attributes:
        _default_features (List[str]): The default set of features to use for computing abstractness.

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

    def _compute_single_pred_single_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Computes the abstractness score for a single predicted text and a single reference text.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference text.
            reduce_fn (Callable): A function to use for reducing the abstractness scores across multiple examples.
            n (int): The size of the n-grams to use for computing abstractness.

        Returns:
            Dict[str, float]: A dictionary containing the abstractness score.
        """
        result = self.__compute_abstractness(references, predictions, n)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Computes the abstractness score for a single predicted text and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the abstractness scores across multiple examples.
            n (int): The size of the n-grams to use for computing abstractness.

        Returns:
            Dict[str, float]: A dictionary containing the abstractness score.
        """
        result = self.__compute_abstractness(references, predictions, n)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Computes the abstractness score for multiple predicted texts and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted texts.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the abstractness scores across multiple examples.
            n (int): The size of the n-grams to use for computing abstractness.

        Returns:
            Dict[str, float]: A dictionary containing the abstractness score.
        """
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
        """
        Computes the abstractness score for a set of reference and predicted texts.

        Args:
            res_references (List[str]): A list of reference texts.
            res_predictions (List[str]): A list of predicted texts.
            n (int): The size of the n-grams to use for computing abstractness.

        Returns:
            float: The abstractness score for the predicted texts.
        """
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
