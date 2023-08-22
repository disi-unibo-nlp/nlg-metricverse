# coding=utf-8

""" Abstractness metric. """

import evaluate
from typing import Callable
from nltk import ngrams

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration

_CITATION = """\
@inproceedings{durmus-etal-2020-feqa,
    title = "{FEQA}: A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization",
    author = "Durmus, Esin  and
      He, He  and
      Diab, Mona",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.454",
    doi = "10.18653/v1/2020.acl-main.454",
    pages = "5055--5070",
    abstract = "Neural abstractive summarization models are prone to generate content inconsistent with the source document, i.e. unfaithful. Existing automatic metrics do not capture such mistakes effectively. We tackle the problem of evaluating faithfulness of a generated summary given its source document. We first collected human annotations of faithfulness for outputs from numerous models on two datasets. We find that current models exhibit a trade-off between abstractiveness and faithfulness: outputs with less word overlap with the source document are more likely to be unfaithful. Next, we propose an automatic question answering (QA) based metric for faithfulness, FEQA, which leverages recent advances in reading comprehension. Given question-answer pairs generated from the summary, a QA model extracts answers from the document; non-matched answers indicate unfaithful information in the summary. Among metrics based on word overlap, embedding similarity, and learned language understanding models, our QA-based metric has significantly higher correlation with human faithfulness scores, especially on highly abstractive summaries.",
}

"""

_DESCRIPTION = """ A metric for measuring the abstractness of generated text. This metric computes the proportion of n-grams in the generated text that do not appear in the reference text. The abstractness score ranges from 0 to 1, with higher values indicating more abstract text.

    Attributes:
        _default_features (List[str]): The default set of features to use for computing abstractness.

"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: List of predictions to score. Each prediction should be a string.
    references: List of references for each prediction. Each reference should be a string.
    n (int): The size of the n-grams to use for computing abstractness. Defaults to 1.

Returns:
    score: The abstractness score for the predicted texts.
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
        Computes the abstractness score for a single predicted text and a single reference text.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference text.
            reduce_fn (Callable): A function to use for reducing the abstractness scores across multiple examples.
            n (int): The size of the n-grams to use for computing abstractness.
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
