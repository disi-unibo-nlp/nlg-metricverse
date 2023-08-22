# coding=utf-8
# Copyright 2021 Open Business Software Solutions, The HuggingFace evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ChrF(++) (Character n-gram F-score) metric. The part of this file is adapted from HuggingFace's
evaluate package implementation of CHRF metric. See
https://github.com/huggingface/evaluate/blob/master/metrics/chrf/chrf.py
"""
from typing import Callable, Dict, List, Tuple, Union

import evaluate
from packaging import version

from nlgmetricverse.collator import Collator
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics import EvaluationInstance, MetricForLanguageGeneration
from nlgmetricverse.metrics._core.utils import PackagePlaceholder, requirement_message

# `import sacrebleu as scb` placeholder
scb = PackagePlaceholder(version="2.0.0")


_CITATION = """\
@inproceedings{popovic-2015-chrf,
    title = "chr{F}: character n-gram {F}-score for automatic {MT} evaluation",
    author = "Popovi{\'c}, Maja",
    booktitle = "Proceedings of the Tenth Workshop on Statistical Machine Translation",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W15-3049",
    doi = "10.18653/v1/W15-3049",
    pages = "392--395",
}
@inproceedings{popovic-2017-chrf,
    title = "chr{F}++: words helping character n-grams",
    author = "Popovi{\'c}, Maja",
    booktitle = "Proceedings of the Second Conference on Machine Translation",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-4770",
    doi = "10.18653/v1/W17-4770",
    pages = "612--618",
}
@inproceedings{post-2018-call,
    title = "A Call for Clarity in Reporting {BLEU} Scores",
    author = "Post, Matt",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
    month = oct,
    year = "2018",
    address = "Belgium, Brussels",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6319",
    pages = "186--191",
}
"""

_DESCRIPTION = """\
ChrF and ChrF++ are two MT evaluation metrics. They both use the F-score statistic for character n-gram matches,
and ChrF++ adds word n-grams as well which correlates more strongly with direct assessment.
ChrF compares character n-grams in the reference and candidate sentences, instead of matching word n-grams as done
in BLEU, ROUGE, etc. The precision and recall are computed over the character n-grams for various values of n (upto 6)
and are combined using arithmetic averaging to get the overall precision (chrP) and recall (chrR) respectively. In
other words, chrP represents the percentage of matched character n-grams present in the hypothesis and chrR represents
the percentage of character n-grams in the reference which are also present in the hypothesis. The final chrF score
is then computed as:
$chrF_{\beta} = (1 + \beta^2) \frac{chrP chrR}{\beta^2 chrP + chrR}$,
where \beta indicates that recall is given \beta times more weightage than precision.
Popovic propose two enhanced versions of chrF: (i) chrF+, which also considers word unigrams; (ii) chrF++, which
considers word unigrams and bigrams in addition to character n-grams.

We use the implementation that is already present in sacreBLEU.
The implementation here is slightly different from sacreBLEU in terms of the required input format. The length of
the references and hypotheses lists need to be the same, so you may need to transpose your references compared to
sacrebleu's required input format. See https://github.com/huggingface/evaluate/issues/3154#issuecomment-950746534.
See the README.md file at https://github.com/mjpost/sacreBLEU#chrf--chrf for more information.
"""

_KWARGS_DESCRIPTION = """
Produces ChrF(++) scores for hypotheses given reference translations.
Args:
    `predictions` (list of str): The system stream (a sequence of segments).
    `references` (list of str): A list of one or more reference streams (each a sequence of segments).
    `char_order` (int): Character n-gram order.
    `word_order` (int): Word n-gram order. If equals to 2, the metric is referred to as chrF++.
    `beta` (int): Determine the importance of recall w.r.t precision.
    `lowercase` (bool): Enable case-insensitivity.
    `whitespace` (bool): If `True`, include whitespaces when extracting character n-grams.
    `eps_smoothing` (bool): If `True`, applies epsilon smoothing similar
         to reference chrF++.py, NLTK and Moses implementations. Otherwise,
         it takes into account effective match order similar to sacreBLEU < 2.0.0.
Returns:
    'score' (float): The chrF (chrF++) score.
    'char_order' (int): The character n-gram order.
    'word_order' (int): The word n-gram order. If equals to 2, the metric is referred to as chrF++.
    'beta' (int): Determine the importance of recall w.r.t precision.
Examples:
    >>> predictions = [
        ["the cat is on the mat", "There is cat playing on the mat"],
        ["Look! a wonderful day.", "There is a good weather outside"]
    ]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> scorer = NLGMetricverse(metrics=load_metric("chrf"))
    >>> scores = scorer(predictions=prediction, references=reference)
    >>> print(scores)
    {'total_items': 2, 'empty_items': 0, 'chrf': {'score': 0.44298405744188873, 'char_order': 6, 'word_order': 0, 'beta': 2}}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CHRFPlanet(MetricForLanguageGeneration):
    def _info(self):
        if version.parse(scb.__version__) < version.parse("1.4.12"):
            raise ImportWarning(
                "To use `sacrebleu`, the module `sacrebleu>=1.4.12` is required, and the current version of "
                "`sacrebleu` doesn't match this condition.\nYou can install it with `pip install sacrebleu>=1.4.12`."
            )
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/mjpost/sacreBLEU#chrf--chrf",
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=1,
            lower_bound=0,
            features=self._default_features,
            codebase_urls=["https://github.com/mjpost/sacreBLEU#chrf--chrf"],
            reference_urls=[
                "https://github.com/m-popovic/chrF",
            ],
        )

    def _download_and_prepare(self, dl_manager):
        global scb
        global CHRFScorer
        try:
            from sacrebleu import CHRF as CHRFScorer
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="chrf", package_name="sacrebleu"))
        else:
            super(CHRFPlanet, self)._download_and_prepare(dl_manager)

    @staticmethod
    def _validate_references(references: Collator) -> None:
        """
        The purpose of this method is to validate the references for a given prediction using the SacreBLEU metric.
        The method first determines the number of references per prediction by getting the length of the first reference in the
        references parameter. It then checks if the length of each reference in the references parameter is equal to the 
        references_per_prediction value.
        """
        references_per_prediction = len(references[0])
        if any(len(refs) != references_per_prediction for refs in references):
            raise ValueError("Sacrebleu requires the same number of references for each prediction")

    @staticmethod
    def _compute_chrf_score(
            predictions: Union[str, List[str]], references: Union[str, List[str]], **kwargs
    ) -> Tuple[float, int, int, int]:
        """
        This function takes in two parameters: predictions and references, both of which can be either a string or a list of strings. 
        The function also accepts additional keyword arguments that can be used to configure the CHRF scorer.
        """
        if kwargs.get("char_order") is None:
            kwargs["char_order"] = CHRFScorer.CHAR_ORDER
        if kwargs.get("word_order") is None:
            kwargs["word_order"] = CHRFScorer.WORD_ORDER
        if kwargs.get("beta") is None:
            kwargs["beta"] = CHRFScorer.BETA
        sb_chrf = CHRFScorer(**kwargs)
        output = sb_chrf.corpus_score(predictions, references)

        return (
            output.score / 100,
            output.char_order,
            output.word_order,
            output.beta,
        )

    def _compute_single_pred_single_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        reduce_fn: Callable = None,
        char_order: int = None,
        word_order: int = None,
        beta: int = None,
        lowercase: bool = False,
        whitespace: bool = False,
        eps_smoothing: bool = False,
    ):
        """
        Compute the chrf score for a single prediction and a single reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a single text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            char_order (int, optional): Character n-gram order.
            word_order (int, optional): Word n-gram order. If equals to 2, the metric is referred to as chrF++.
            beta (int, optional): Determine the importance of recall w.r.t precision.
            lowercase (bool, optional): Enable case-insensitivity.
            whitespace (bool, optional): If `True`, include whitespaces when extracting character n-grams.
            eps_smoothing (bool, optional): If `True`, applies epsilon smoothing similar
        """
        score, c_ord, w_ord, beta = self._compute_chrf_score(
            predictions,
            references,
            char_order=char_order,
            word_order=word_order,
            beta=beta,
            lowercase=lowercase,
            whitespace=whitespace,
            eps_smoothing=eps_smoothing,
        )
        return {"score": score, "char_order": c_ord, "word_order": w_ord, "beta": beta}

    def _compute_single_pred_multi_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        reduce_fn: Callable = None,
        char_order: int = None,
        word_order: int = None,
        beta: int = None,
        lowercase: bool = False,
        whitespace: bool = False,
        eps_smoothing: bool = False,
    ):
        """
        Compute the chrf score for a single prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            char_order (int, optional): Character n-gram order.
            word_order (int, optional): Word n-gram order. If equals to 2, the metric is referred to as chrF++.
            beta (int, optional): Determine the importance of recall w.r.t precision.
            lowercase (bool, optional): Enable case-insensitivity.
            whitespace (bool, optional): If `True`, include whitespaces when extracting character n-grams.
            eps_smoothing (bool, optional): If `True`, applies epsilon smoothing similar
        """
        # SacreBleu inherently supports multiple references.
        return self._compute_single_pred_single_ref(
            predictions=predictions,
            references=references,
            reduce_fn=reduce_fn,
            char_order=char_order,
            word_order=word_order,
            beta=beta,
            lowercase=lowercase,
            whitespace=whitespace,
            eps_smoothing=eps_smoothing,
        )

    def _compute_multi_pred_multi_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        reduce_fn: Callable = None,
        char_order: int = None,
        word_order: int = None,
        beta: int = None,
        lowercase: bool = False,
        whitespace: bool = False,
        eps_smoothing: bool = False,
    ):
        """
        Compute the chrf score for multiple prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a multiple text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            char_order (int, optional): Character n-gram order.
            word_order (int, optional): Word n-gram order. If equals to 2, the metric is referred to as chrF++.
            beta (int, optional): Determine the importance of recall w.r.t precision.
            lowercase (bool, optional): Enable case-insensitivity.
            whitespace (bool, optional): If `True`, include whitespaces when extracting character n-grams.
            eps_smoothing (bool, optional): If `True`, applies epsilon smoothing similar
        """
        scores = []
        for preds, refs in zip(predictions, references):
            pred_scores = []
            for pred in preds:
                score, _, _, _ = self._compute_chrf_score(
                    pred,
                    refs,
                    char_order=char_order,
                    word_order=word_order,
                    beta=beta,
                    lowercase=lowercase,
                    whitespace=whitespace,
                    eps_smoothing=eps_smoothing,
                )
                pred_scores.append(score)
            scores.append(float(reduce_fn(pred_scores)))

        return {
            "score": sum(scores) / len(scores),
            "char_order": char_order or CHRFScorer.CHAR_ORDER,
            "word_order": word_order or CHRFScorer.WORD_ORDER,
            "beta": beta or CHRFScorer.BETA,
        }

    def evaluate(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ) -> Dict[str, float]:
        """
        This function takes in three parameters: predictions, references, and reduce_fn, as well as additional keyword arguments. 
        The purpose of this method is to evaluate the predictions against the references using the metric implemented by the class.
        """
        if predictions.can_collapse() and references.can_collapse():
            predictions = predictions.collapse()
            eval_fn = self._compute_single_pred_single_ref
        elif predictions.can_collapse() and not references.can_collapse():
            predictions = predictions.collapse()
            eval_fn = self._compute_single_pred_multi_ref
        else:
            eval_fn = self._compute_multi_pred_multi_ref
        self._validate_references(references)
        return eval_fn(predictions=predictions, references=references, reduce_fn=reduce_fn, **kwargs)
