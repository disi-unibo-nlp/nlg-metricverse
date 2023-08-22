# coding=utf-8

""" Compression metric. """

import evaluate
from typing import Callable
from nltk import word_tokenize
from collections import namedtuple as _namedtuple
from tqdm import tqdm

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration

Match = _namedtuple("Match", ("summary", "text", "length"))

_CITATION = """

"""

_DESCRIPTION = """
The Compression metric calculates the documention-summary word ration: |x|/|y|
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: List of predictions to score. Each prediction should be a string.
    references: List of references for each prediction. Each reference should be a string.
Returns:
    compression: The average compression of the predictions.
Examples:
    >>> scorer = NLGMetricverse(metrics=load_metric("compression"))
    >>> predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
    >>> references = ["Peace at home, peace in th world.", "The cat is playing on the mat."]
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    { "total_items": 2, "empty_items": 0, "compression": { "score": 0.95 }}
"""

_LICENSE = """

"""

CHECKPOINT_URLS = {

}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CompressionPlanet(MetricForLanguageGeneration):
    def _info(self):
        """
        Returns metadata about the metric.

        Returns:
            MetricInfo: An object containing metadata about the metric.
        """
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=1,
            lower_bound=0,
            features=self._default_features,
            codebase_urls=[""],
            reference_urls=[
                ""
            ],
        )

    def _compute_single_pred_single_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the compression score for a single predicted text and a single reference text.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference text.
            reduce_fn (Callable): A function to use for reducing the compression scores across multiple examples.
        """
        result = self._compute_compression(references, predictions)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the compression score for a single predicted text and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the compression scores across multiple examples.
        """
        predList = [str(pred) for pred in predictions]
        refList = [ref for refs in references for ref in refs]

        result = self._compute_compression(refList, predList)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the compression score for multiple predicted texts and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted texts.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the compression scores across multiple examples.
        """
        predList = []
        refList = []
        for pred in predictions:
            predList += pred
        for ref in references:
            refList += ref
        result = self._compute_compression(refList, predList)
        return {"score": result}

    @staticmethod
    def _compute_compression(references, predictions):
        """
        The function computes the compression ratio between the reference and predicted texts. Firstly it is initialized an empty 
        list tot_compression. It then iterates over the indices of the references list using the range function and the len 
        function. For each index i, the function tokenizes the references[i] and predictions[i] strings using the
        word_tokenize function from the nltk library. If both the words_source and words_target lists have a length 
        greater than 0, the function computes the compression ratio by dividing the length of words_source by the length 
        of words_target. The resulting value is appended to the tot_compression list.
        Finally, the function computes the average compression ratio by dividing the sum of the values in tot_compression 
        by the length of tot_compression. The resulting value is rounded to two decimal places using the round function and 
        returned as the output of the function.
        """
        tot_compression = []
        for i in tqdm(range(min(len(references), len(predictions)))):
            words_source = word_tokenize(references[i])
            words_target = word_tokenize(predictions[i])
            if len(words_source) > 0 and len(words_target) > 0:
                tot_compression.append(len(words_source) / len(words_target))
        avg_compression = round(sum(tot_compression) / len(tot_compression), 2)
        return avg_compression