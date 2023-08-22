# coding=utf-8

""" Average Unique N-Gram metric. """

import evaluate
from typing import Callable
from nltk import ngrams

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration
from nlgmetricverse.utils.data_structure import remove_duplicates

_CITATION = """\
@inproceedings{xiao-carenini-2020-systematically,
    title = "Systematically Exploring Redundancy Reduction in Summarizing Long Documents",
    author = "Xiao, Wen  and
      Carenini, Giuseppe",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.aacl-main.51",
    pages = "516--528",
    abstract = "Our analysis of large summarization datasets indicates that redundancy is a very serious problem when summarizing long documents. Yet, redundancy reduction has not been thoroughly investigated in neural summarization. In this work, we systematically explore and compare different ways to deal with redundancy when summarizing long documents. Specifically, we organize existing methods into categories based on when and how the redundancy is considered. Then, in the context of these categories, we propose three additional methods balancing non-redundancy and importance in a general and flexible way. In a series of experiments, we show that our proposed methods achieve the state-of-the-art with respect to ROUGE scores on two scientific paper datasets, Pubmed and arXiv, while reducing redundancy significantly.",
}
"""

_DESCRIPTION = """\
    A class for computing the Abstractness Universe (AUN) metric for generated text.

    The AUN metric measures the proportion of n-grams in the generated text that do not appear in the reference text.
    The metric ranges from 0 to 1, with higher values indicating more abstract text.

    Args:
        n (int): The size of the n-grams to use for computing abstractness. Defaults to 1 (unigrams).

"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: An instance of EvaluationInstance containing the predicted text.
    references: An instance of EvaluationInstance containing the reference text.

Returns:
    'score': Aun score.
Examples:
    >>> score = nlgmetricverse.load_metric("accuracy")
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = score.compute(predictions=predictions, references=references)
    >>> print(results)
    {'aun': {'score': 0.9310344827586207}}
"""

_LICENSE = """

"""

CHECKPOINT_URLS = {

}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AUNPlanet(MetricForLanguageGeneration):
    def _info(self):
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
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Computes the AUN metric for a single predicted text and a single reference text.

        Args:
            predictions (EvaluationInstance): An instance of EvaluationInstance containing the predicted text.
            references (EvaluationInstance): An instance of EvaluationInstance containing the reference text.

        Returns:
            dict: A dictionary containing the AUN score.
        """
        result = self.__compute_average_unique_ngram(predictions, n)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Computes the AUN metric for a single predicted text and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An instance of EvaluationInstance containing the predicted text.
            references (EvaluationInstance): An instance of EvaluationInstance containing the reference texts.

        Returns:
            dict: A dictionary containing the AUN score.
        """
        result = self.__compute_average_unique_ngram(predictions, n)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, n=1
    ):
        """
        Computes the AUN metric for multiple predicted texts and multiple reference texts.

        Args:
            predictions (EvaluationInstance): A list of EvaluationInstance instances containing the predicted texts.
            references (EvaluationInstance): A list of EvaluationInstance instances containing the reference texts.
            
        Returns:
            dict: A dictionary containing the AUN score.
        """
        inputList = []
        for prediction in predictions:
            inputList += prediction
        result = self.__compute_average_unique_ngram(inputList, n)
        return {"score": result}

    @staticmethod
    def __compute_average_unique_ngram(predictions, n):
        """
        Computes the average proportion of unique n-grams in a set of predicted texts.

        Args:
            predictions (list): A list of strings containing the predicted texts.
            n (int): The size of the n-grams to use for computing abstractness.

        Returns:
            float: The average proportion of unique n-grams across all predicted texts.
        """
        n_grams_count = 0
        unique_n_grams_count = 0

        for candidate in predictions:
            n_grams = list(ngrams(candidate.split(), n))
            for _ in n_grams:
                n_grams_count += 1
            unique_n_grams = remove_duplicates(n_grams)
            unique_n_grams_count += len(unique_n_grams)
        return unique_n_grams_count / n_grams_count