# coding=utf-8

""" Density metric. """

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
The Density metric measures the average lenght of the extractive fragments. 
it is formulated as: \frac{1}{|y|} \sum_{f \in F(x,y)} (|f|_c)^2}
  
Where ||_c is the character lenght. When low, it suggest that most summary sentences are not 
verbatim extractions from the sources (abstractive).
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: List of predictions to score. Each prediction should be a string.
    references: List of references for each prediction. Each reference should be a string.
Returns:
    density: The average density of the predictions.
Examples:
    >>> scorer = NLGMetricverse(metrics=load_metric("density"))
    >>> predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
    >>> references = ["Peace at home, peace in th world.", "The cat is playing on the mat."]
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    { "total_items": 2, "empty_items": 0, "density": { "score": 1.97 }}
"""

_LICENSE = """

"""

CHECKPOINT_URLS = {

}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class DensityPlanet(MetricForLanguageGeneration):
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
            upper_bound=10,
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
        Computes the density score for a single predicted text and a single reference text.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference text.
            reduce_fn (Callable): A function to use for reducing the density scores across multiple examples.
        """
        result = self._compute_density(references, predictions)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the density score for a single predicted text and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the density scores across multiple examples.
        """
        predList = [str(pred) for pred in predictions]
        refList = [ref for refs in references for ref in refs]

        result = self._compute_density(refList, predList)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the density score for multiple predicted texts and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted texts.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the density scores across multiple examples.
        """
        predList = []
        refList = []
        for pred in predictions:
            predList += pred
        for ref in references:
            refList += ref
        result = self._compute_density(refList, predList)
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
        density = sum(o.length ** 2 for o in matches) / len(a)
        return density

    @staticmethod
    def _compute_density(references, predictions):
        """
        The method that calculates a density metric for sets of text references and predictions. 
        It processes each reference-prediction pair, tokenizes the words in both sets, and ensures that they 
        are non-empty. The words are then normalized to lowercase. The normalized word lists are used to compute 
        a density score through the `DensityPlanet.match_texts` function. The density scores for all pairs 
        are collected, and an average density is computed by summing these scores and dividing by the total
        number of pairs. The resulting average density is rounded to two decimal places and returned by the function.
        """
        tot_density = []
        tot_fragments = []
        for i in tqdm(range(min(len(references), len(predictions)))):
            words_source = word_tokenize(references[i])
            words_target = word_tokenize(predictions[i])
            if len(words_source) > 0 and len(words_target) > 0:
                words_source_norm = [str(t).lower() for t in words_source]
                words_target_norm = [str(t).lower() for t in words_target]
                density = DensityPlanet.match_texts(words_target_norm, words_source_norm)
                tot_density.append(density)
        avg_density = round(sum(tot_density) / len(tot_density), 2)
        return avg_density