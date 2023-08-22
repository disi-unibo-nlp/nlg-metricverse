# coding=utf-8

""" Coverage metric. """

import evaluate
import numpy as np
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
The Coverage metric measures the percentage of summary words within the source text: 
\frac{1}{|y|} \sum_{f \in F(x,y)} |f|}

Where F is the set of all fragments, i.e., extractive character sequences. When low, it suggest a 
high change for unsupported entities and facts.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: List of predictions to score. Each prediction should be a string.
    references: List of references for each prediction. Each reference should be a string.
Returns:
    coverage: The average coverage of the predictions.
Examples:
    >>> scorer = NLGMetricverse(metrics=load_metric("coverage"))
    >>> predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
    >>> references = ["Peace at home, peace in th world.", "The cat is playing on the mat."]
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    { "total_items": 2, "empty_items": 0, "coverage": { "score": 0.77 }}
"""

_LICENSE = """

"""

CHECKPOINT_URLS = {

}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CoveragePlanet(MetricForLanguageGeneration):
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
        Computes the coverage score for a single predicted text and a single reference text.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference text.
            reduce_fn (Callable): A function to use for reducing the coverage scores across multiple examples.
        """
        result = self._compute_coverage(references, predictions)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the coverage score for a single predicted text and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the coverage scores across multiple examples.
        """
        predList = [str(pred) for pred in predictions]
        refList = [ref for refs in references for ref in refs]

        result = self._compute_coverage(refList, predList)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the coverage score for multiple predicted texts and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted texts.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the coverage scores across multiple examples.
        """
        predList = []
        refList = []
        for pred in predictions:
            predList += pred
        for ref in references:
            refList += ref
        result = self._compute_coverage(refList, predList)
        return {"score": result}

    @staticmethod
    def match_texts(a, b):
        matches = []
        a_start = 0
        b_start = 0
        while a_start < len(a):
            best_match = None
            best_match_length = 0
            while b_start < len(b):
                if a[a_start] == b[b_start]:
                    a_end = a_start
                    b_end = b_start
                    while a_end < len(a) and b_end < len(b) and b[b_end] == a[a_end]:
                        b_end += 1
                        a_end += 1
                    length = a_end - a_start
                    if length > best_match_length:
                        best_match = Match(a_start, b_start, length)
                        best_match_length = length
                    b_start = b_end
                else:
                    b_start += 1
            b_start = 0
            if best_match:
                if best_match_length > 0:
                    matches.append(best_match)
                a_start += best_match_length
            else:
                a_start += 1
        coverage = sum(o.length for o in matches) / len(a)
        return coverage

    @staticmethod
    def _compute_coverage(references, predictions):
        """
        The method processes two sets of text references and predictions to compute a coverage metric. 
        It begins by initializing an empty list named `tot_coverage` to hold coverage values for each 
        reference-prediction pair. Utilizing the `tqdm` library for progress tracking, the method iterates 
        through these pairs. It tokenizes the reference and prediction texts into words, verifying non-empty 
        word counts before proceeding to lowercase normalization. Using the `CoveragePlanet.match_texts` function, 
        it calculates a coverage score by comparing normalized word lists. This score signifies the alignment 
        between prediction and reference words. The coverage score is accumulated within the `tot_coverage` list. 
        After iterating through all pairs, the method determines average coverage by summing coverage values and 
        dividing by the total pair count. The average coverage is then rounded to two decimals. Ultimately, the
        method yields the computed average coverage as its output.
        """
        tot_coverage = []
        for i in tqdm(range(min(len(references), len(predictions)))):
            words_source = word_tokenize(references[i])
            words_target = word_tokenize(predictions[i])
            if len(words_source) > 0 and len(words_target) > 0:
                words_source_norm = [str(t).lower() for t in words_source]
                words_target_norm = [str(t).lower() for t in words_target]
                coverage = CoveragePlanet.match_texts(words_source_norm, words_target_norm)
                tot_coverage.append(coverage)
        avg_coverage = round(sum(tot_coverage) / len(tot_coverage), 2)
        return avg_coverage