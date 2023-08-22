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
BLEU and ORANGE (smooth-BLEU, i.e., sentBLEU) metrics. The part of this file is adapted from HuggingFace's
evaluate package implementation of BLEU metric. See
https://github.com/huggingface/evaluate/blob/master/metrics/bleu/bleu.py
"""

import math
from typing import Callable, Dict, Tuple

import datasets
import evaluate

from nlgmetricverse.collator import Collator
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration
from nlgmetricverse.metrics._core.utils import get_token_lengths
from nlgmetricverse.tokenizer import DefaultTokenizer, TokenizerWrapper

_CITATION = """\
@inproceedings{PapineniRWZ02,
  author    = {Kishore Papineni and
               Salim Roukos and
               Todd Ward and
               Wei{-}Jing Zhu},
  title     = {Bleu: a Method for Automatic Evaluation of Machine Translation},
  booktitle = {Proceedings of the 40th Annual Meeting of the Association for Computational
               Linguistics, July 6-12, 2002, Philadelphia, PA, {USA}},
  pages     = {311--318},
  publisher = {{ACL}},
  year      = {2002},
  url       = {https://aclanthology.org/P02-1040/},
  doi       = {10.3115/1073083.1073135}
}
@inproceedings{lin-och-2004-orange,
    title = "{ORANGE}: a Method for Evaluating Automatic Evaluation Metrics for Machine Translation",
    author = "Lin, Chin-Yew  and
      Och, Franz Josef",
    booktitle = "{COLING} 2004: Proceedings of the 20th International Conference on Computational Linguistics",
    month = "aug 23{--}aug 27",
    year = "2004",
    address = "Geneva, Switzerland",
    publisher = "COLING",
    url = "https://www.aclweb.org/anthology/C04-1072",
    pages = "501--507",
}
"""

_DESCRIPTION = """
BLEU (bilingual evaluation understudy) scores were originally developed in the context of machine translation, but
they are applied in other generation tasks as well. Quality is considered to be the correspondence between a
machine's output and that of a human: "the closer a machine translation is to a professional human translation, the
better it is" – this is the central idea behind BLEU. BLEU was one of the first metrics to claim a high correlation
with human judgements of quality, and remains one of the most popular automated and inexpensive metrics.
For BLEU scoring, we require a dataset $Y$ consisting of instances $(a, B)$ where $a$ is a candidate (a model
prediction) and $B$ is a set of gold texts. The metric has two main components.
- Modified n-gram precision. A direct application of precision would divide the number of correct n-grams in the
  candidate (n-grams that appear in any translation) by the total number of n-grams in the candidate. This has a
  degenerate solution in which the predicted output contains only one n-gram. BLEU's modified version substitutes
  the actual count for each n-gram s in the candidate by the maximum number of times s appears in any gold text.
- Brevity penalty (BP). To avoid favoring outputs that are too short, a penalty is applied. Let $r$ be the sum of all
  minimal absolute length differences between candidates and referents in the dataset $Y$, and let $c$ be the sum of
  the lengths of all the candidates. Then:
  $BP(Y) = \begin{cases} 1 & \textrm{ if } c > r \\ \exp(1 - \frac{r}{c}) & \textrm{otherwise}\end{cases}$
The BLEU score itself is typically a combination of modified n-gram precision for various n (usually up to 4):
$BLEU(Y) = BP(Y) \cdot \exp\left(\sum_{n=1}^{N} w_{n} \cdot \log\left(modified-precision(Y, n\right)\right)$
where Y is the dataset, and w_n is a weight for each n-gram level (usually set to 1/n).
By definition, BLEU is a corpus-level metric, since the statistics above are computed across sentences over an entire
test set. The sentence-level variant requires a smoothing strategy to counteract the effect of 0 n-gram precisions,
which are more probable with shorter texts.
Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good 
quality reference translations. Those scores are then averaged over the whole corpus to reach an estimate of the 
translation's overall quality.
It has many affinities with WER, but seeks to accommodate the fact that there are typically multiple suitable outputs
for a given input.
"""

_KWARGS_DESCRIPTION = """
Computes BLEU score of translated segments against one or more references.
Args:
    predictions: list of translations to score.
        Each translation should be tokenized into a list of tokens.
    references: list of lists of references for each translation.
        Each reference should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
Returns:
    'score': BLEU score,
    'precisions': geometric mean of n-gram precisions,
    'brevity_penalty': brevity penalty,
    'length_ratio': ratio of lengths,
    'translation_length': translation_length,
    'reference_length': reference_length
Examples:
    >>> scorer = NLGMetricverse(metrics=load_metric("bleu"))
    >>> predictions = [
        ["the cat is on the mat", "There is cat playing on the mat"],
        ["Look! a wonderful day.", "There is a good weather outside"]
    ]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    {'total_items': 2, 'empty_items': 0, 'bleu': {'score': 0.3378703280802838,
    'precisions': [0.84, 0.5714285714285714, 0.35294117647058826, 0.07692307692307693],
    'brevity_penalty': 1.0, 'length_ratio': 1.1818181818181819,
    'translation_length': 13, 'reference_length': 11}}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BleuPlanet(MetricForLanguageGeneration):
    def __init__(self, resulting_name: str = None, compute_kwargs: Dict = None, **kwargs):
        self.should_change_resulting_name = resulting_name is None
        self.tokenizer = DefaultTokenizer()
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _info(self):
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=1,
            lower_bound=0,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string", id="tokens"), id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="tokens"), id="sequence"),
                }
            ),
            codebase_urls=[
                "https://github.com/tensorflow/nmt/blob/0be864257a76c151eef20ea689755f08bc1faf4e/nmt/scripts/bleu.py"
            ],
            reference_urls=[
                "https://en.wikipedia.org/wiki/BLEU",
                "https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213",
            ],
        )

    def _download_and_prepare(self, dl_manager) -> None:
        """
        Downloads and import the computation of bleu score from the implementation
        of BLEU computation from tensorflow/nmt. The code is sourced from a specific
        commit on the master branch, in order to keep things stable. See
        https://github.com/tensorflow/nmt/blob/0be864257a76c151eef20ea689755f08bc1faf4e/nmt/scripts/bleu.py
        """
        nmt_source = "https://raw.githubusercontent.com/tensorflow/nmt/0be864257a76c151eef20ea689755f08bc1faf4e/nmt/scripts/bleu.py"
        self.external_module_path = dl_manager.download(nmt_source)

    def _preprocess(self, predictions: Collator, references: Collator) -> Tuple[Collator, Collator]:
        tokenizer_wrapper = TokenizerWrapper(self.tokenizer)
        return tokenizer_wrapper.tokenize(predictions, references)

    def _compute_bleu_score(self, predictions: Collator, references: Collator, max_order=4, smooth=False):
        evaluation_fn = self._get_external_resource("nmt_bleu", attr="compute_bleu")
        score = evaluation_fn(
            reference_corpus=references, translation_corpus=predictions, max_order=max_order, smooth=smooth
        )
        (bleu, precisions, bp, ratio, translation_length, reference_length) = score
        return {
            "score": bleu,
            "precisions": precisions,
            "brevity_penalty": bp,
            "length_ratio": ratio,
            "translation_length": translation_length,
            "reference_length": reference_length,
        }

    def _compute_single_pred_single_ref(
        self,
        predictions: Collator,
        references: Collator,
        reduce_fn: Callable = None,
        max_order: int = 4,
        smooth: bool = False,
    ):
        # _compute_bleu_score expects a slightly different reference structure
        #   than the one we currently have for single prediction single reference
        references = [[r] for r in references]
        return self._compute_bleu_score(
            predictions=predictions, references=references, max_order=max_order, smooth=smooth
        )

    def _compute_single_pred_multi_ref(
        self,
        predictions: Collator,
        references: Collator,
        reduce_fn: Callable = None,
        max_order: int = 4,
        smooth: bool = False,
    ):
        # Bleu score inherently supports multiple references.
        # Bypassing _compute_single_pred_single_ref, as it does structure
        #   manipulation that isn't needed for here
        return self._compute_bleu_score(
            predictions=predictions, references=references, max_order=max_order, smooth=smooth
        )

    def _compute_multi_pred_multi_ref(
        self,
        predictions: Collator,
        references: Collator,
        reduce_fn: Callable = None,
        max_order: int = 4,
        smooth: bool = False,
    ):
        """
        For multiple predictions in the input, the resulting bleu score is computed using corpus level bleu
        as if it is the same for single prediction inputs (which is directly supported by evaluation function).
        The only difference however is that, since the score is calculated corpus level, `reduce_fn` is ignored.
        In other words, there is no aggregation for multiple predictions, and it is rather handled by the corpus
        level bleu score computation. After the pure blue score is calculated, parameters are changed accordingly,
        so the resulting predictions may not be a direct match of bleu score equivalent.
        Args:
            predictions: (Collator) Collator object of predictions.
            references: (Collator) Collator object of references.
            reduce_fn: (Callable) Ignored.
            max_order: (int) Maximum order of bleu precision counts (e.g n in BLEU-n), passed to evaluation function.
            smooth: (bool) Whether to apply smoothing or not, passed to evaluation function. False by default.
        """
        flattened_predictions = []
        matched_references = []
        adjusted_reference_length = adjusted_prediction_length = 0
        for preds, refs in zip(predictions, references):
            n_preds = len(preds)
            adjusted_reference_length += get_token_lengths(refs, reduce_fn=min)
            adjusted_prediction_length += get_token_lengths(preds, reduce_fn=max)
            flattened_predictions.extend([pred for pred in preds])
            matched_references.extend([refs] * n_preds)

        flattened_predictions = Collator(flattened_predictions, keep=True)
        matched_references = Collator(matched_references, keep=True)
        score = self._compute_single_pred_multi_ref(
            predictions=flattened_predictions,
            references=matched_references,
            reduce_fn=reduce_fn,
            max_order=max_order,
            smooth=smooth,
        )

        prediction_length, reference_length = score["translation_length"], score["reference_length"]
        ratio = prediction_length / reference_length
        adjusted_ratio = adjusted_prediction_length / adjusted_reference_length

        bleu_score = score["score"]
        if ratio > 1.0:
            adjusted_bp = 1.0
        else:
            bp = math.exp(1 - 1.0 / ratio)
            adjusted_bp = math.exp(1 - 1.0 / adjusted_ratio)
            bleu_score = bleu_score * (adjusted_bp / bp)

        score.update(
            {
                "score": bleu_score,
                "precisions": score["precisions"],
                "brevity_penalty": adjusted_bp,
                "length_ratio": adjusted_ratio,
                "translation_length": adjusted_prediction_length,
                "reference_length": adjusted_reference_length,
            }
        )

        return score

    def evaluate(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ) -> Dict[str, float]:
        max_order = kwargs.get("max_order")
        if max_order is not None and self.should_change_resulting_name:
            self.resulting_name += f"_{max_order}"

        return super().evaluate(predictions=predictions, references=references, reduce_fn=reduce_fn, **kwargs)
