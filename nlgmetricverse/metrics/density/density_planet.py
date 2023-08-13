# coding=utf-8

""" Abstractness metric. """

import evaluate
from typing import Callable
from nltk import ngrams
from nltk import word_tokenize
from collections import namedtuple as _namedtuple
from tqdm import tqdm

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.metrics._core import MetricForLanguageGeneration

Match = _namedtuple("Match", ("summary", "text", "length"))

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
class DensityPlanet(MetricForLanguageGeneration):
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
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the coverage score for a single predicted text and a single reference text.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference text.
            reduce_fn (Callable): A function to use for reducing the abstractness scores across multiple examples.
        """
        result = self._compute_density(references, predictions)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the abstractness score for a single predicted text and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the abstractness scores across multiple examples.
        """
        result = self._compute_density(references, predictions)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the abstractness score for multiple predicted texts and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted texts.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the abstractness scores across multiple examples.
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
    def _compute_density(sources, targets):
        tot_density = []
        for i in tqdm(range(len(sources))):
            words_source = word_tokenize(sources[i])
            words_target = word_tokenize(targets[i])
            if len(words_source) > 0 and len(words_target) > 0:
                words_source_norm = [str(t).lower() for t in words_source]
                words_target_norm = [str(t).lower() for t in words_target]
                density = DensityPlanet.match_texts(words_source_norm, words_target_norm)
                tot_density.append(density)
        avg_density = round(sum(tot_density) / len(tot_density), 2)
        return avg_density