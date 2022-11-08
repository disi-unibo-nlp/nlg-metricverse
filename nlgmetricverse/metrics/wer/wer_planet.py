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
""" Word Error Ratio (WER) metric. The part of this file is adapted from HuggingFace's
evaluate package implementation of CER metric. See
https://github.com/huggingface/evaluate/blob/master/metrics/wer/wer.py
"""
import warnings
from typing import Callable, List, Tuple, Union

import evaluate

from nlgmetricverse.metrics import EvaluationInstance, MetricForLanguageGeneration
from nlgmetricverse.metrics._core.utils import PackagePlaceholder, requirement_message

# `import jiwer` placeholder
jiwer = PackagePlaceholder(version="2.3.0")


_CITATION = """\
@inproceedings{MorrisMG04,
  author    = {Andrew Cameron Morris and
               Viktoria Maier and
               Phil D. Green},
  title     = {From {WER} and {RIL} to {MER} and {WIL:} improved evaluation measures
               for connected speech recognition},
  booktitle = {{INTERSPEECH} 2004 - ICSLP, 8th International Conference on Spoken
               Language Processing, Jeju Island, Korea, October 4-8, 2004},
  publisher = {{ISCA}},
  year      = {2004},
  url       = {http://www.isca-speech.org/archive/interspeech\_2004/i04\_2765.html}
}
"""

_DESCRIPTION = """\
The word error rate (WER) metric is a word-level, length-normalized measure of Levenshtein string-edit distance.
It is a common metric for evaluating automatic speech recognition or machine translation systems.
The general difficulty of measuring performance lies in the fact that the recognized word sequence can have a different 
length from the reference word sequence (supposedly the correct one). The WER is derived from the Levenshtein distance, 
working at the word level instead of the phoneme level. The WER is a valuable tool for comparing different systems as 
well as for evaluating improvements within one system. This kind of measurement, however, provides no details on the 
nature of translation errors and further work is therefore required to identify the main source(s) of error and to 
focus any research effort.
This problem is solved by first aligning the recognized word sequence with the reference (spoken) word sequence using 
dynamic string alignment. Examination of this issue is seen through a theory called the power law that states the 
correlation between perplexity and word error rate.
Word error rate can then be computed as:
WER = (S + D + I) / N = (S + D + I) / (S + D + C)
where
S is the number of substitutions,
D is the number of deletions,
I is the number of insertions,
C is the number of correct words,
N is the number of words in the reference (N=S+D+C).
E.g., wer(['A', 'B', 'C'], ['A', 'A', 'C']) = 0.3333333333333333
E.g., wer(['A', 'B', 'C', 'D'], ['A', 'A', 'C', 'D']) = 0.25
To calculate this over the entire test-set, one gets the edit-distances for each gold–predicted pair and normalizes
these by the length of all the gold examples, rather than normalizing each case. This gives a single summary value
for the entire set of errors.
corpus-wer = sum(dists) / sum(lenghts)
This method says that our desired notion of closeness or accuracy can be operationalized in terms of the low-level
operations of insertion, deletion, and substitution. The guiding intuition is very much like that of F scores.

BOUNDS
This value indicates the average number of errors per reference word. The lower the value, the better the
performance of the ASR system with a WER of 0 being a perfect score. So, the range-bound of WER is [0, +inf[, where
0 is best. The lack of a finite upper bound derives from the fact that the normalizing constant is given by the true
sequences, and the predicted sequences can differ from them in any conceivable way in principle.

WEAKNESSES
The value encoded reveals a potential weakness in certain domains.
Roughly, the more semantic the task, the less appropriate WER is likely to be.
For example, adding a negation to a sentence will radically change its meaning but incur only a small WER penalty,
whereas passivizing a sentence (Kim won the race → The race was won by Kim) will hardly change its meaning at all
but incur a large WER penalty.
See also Liu et al. 2016 (https://www.aclweb.org/anthology/D16-1230) for similar arguments in the context of
dialogue generation.

PROPERTY
% of insert, delete, replace

CATEGORY
unsupervised; n-gram overlap

TASKS
MT, SR
"""

_KWARGS_DESCRIPTION = """
Compute WER score of transcribed segments against references.
Args:
    references: List of references for each speech input.
    predictions: List of transcriptions to score.
    concatenate_texts (bool, default=False): Whether to concatenate all input texts or compute WER iteratively.
Returns:
    (float): the word error rate
Examples:
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> wer = nlgmetricverse.load_metric("wer")
    >>> wer_score = wer.compute(predictions=predictions, references=references)
    >>> print(wer_score)
    {
      "wer": {
        "score": 1.0,
        "overall": {
          "substitutions": 2.8333333333333335,
          "deletions": 0.5,
          "insertions": 0.16666666666666666,
          "hits": 2.6666666666666665
        }
      }
    }
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class WERPlanet(MetricForLanguageGeneration):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/jitsi/jiwer/"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/Word_error_rate",
                "https://sites.google.com/site/textdigitisation/qualitymeasures/computingerrorrates",
            ],
        )

    def _download_and_prepare(self, dl_manager):
        global jiwer

        try:
            import jiwer
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="WER", package_name="jiwer"))
        else:
            super(WERPlanet, self)._download_and_prepare(dl_manager)

    @staticmethod
    def _compute_wer_score(
            predictions: Union[str, List[str]], references: Union[str, List[str]]
    ) -> Tuple[float, int, int, int, int]:
        measures = jiwer.compute_measures(references, predictions)
        return (
            measures["wer"],
            int(measures["substitutions"]),
            int(measures["deletions"]),
            int(measures["insertions"]),
            int(measures["hits"]),
        )

    def _compute_single_pred_single_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        reduce_fn: Callable = None,
        concatenate_texts: bool = False,
    ):
        if concatenate_texts:
            score, total_substitutions, total_deletions, total_insertions, total_hits = self._compute_wer_score(
                predictions, references
            )
        else:
            incorrect = 0
            total = 0
            total_substitutions = 0
            total_deletions = 0
            total_insertions = 0
            total_hits = 0
            for prediction, reference in zip(predictions, references):
                _, substitutions, deletions, insertions, hits = self._compute_wer_score(prediction, reference)
                total_substitutions += substitutions
                total_deletions += deletions
                total_insertions += insertions
                total_hits += hits
                incorrect += substitutions + deletions + insertions
                total += substitutions + deletions + hits
            score = incorrect / total

        return {
            "score": score,
            "overall": {
                "substitutions": total_substitutions,
                "deletions": total_deletions,
                "insertions": total_insertions,
                "hits": total_hits,
            },
        }

    def _compute_single_pred_multi_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        reduce_fn: Callable = None,
        **kwargs
    ):
        if "concatenate_texts" in kwargs:
            warnings.warn("Option 'concatenate_texts' is only available in single-pred & single-ref setting.")

        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_hits = 0
        total_refs = 0
        scores = []
        for pred, refs in zip(predictions, references):
            pred_scores = []
            for ref in refs:
                score, substitutions, deletions, insertions, hits = self._compute_wer_score(
                    predictions=pred, references=ref
                )
                pred_scores.append(score)
                total_substitutions += substitutions
                total_deletions += deletions
                total_insertions += insertions
                total_hits += hits
                total_refs += 1
            reduced_score = float(reduce_fn(pred_scores))
            scores.append(reduced_score)

        return {
            "score": float(reduce_fn(scores)),
            "overall": {
                "substitutions": total_substitutions / total_refs,
                "deletions": total_deletions / total_refs,
                "insertions": total_insertions / total_refs,
                "hits": total_hits / total_refs,
            },
        }

    def _compute_multi_pred_multi_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        reduce_fn: Callable = None,
        **kwargs
    ):
        if "concatenate_texts" in kwargs:
            warnings.warn("Option 'concatenate_texts' is only available in single-pred & single-ref setting.")

        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_hits = 0
        total_refs = 0
        scores = []
        for preds, refs in zip(predictions, references):
            pred_scores = []
            for pred in preds:
                for ref in refs:
                    score, substitutions, deletions, insertions, hits = self._compute_wer_score(
                        predictions=pred, references=ref
                    )
                    pred_scores.append(score)
                    total_substitutions += substitutions
                    total_deletions += deletions
                    total_insertions += insertions
                    total_hits += hits
                    total_refs += 1
            reduced_score = float(reduce_fn(pred_scores))
            scores.append(reduced_score)

        return {
            "score": float(reduce_fn(scores)),
            "overall": {
                "substitutions": total_substitutions / total_refs,
                "deletions": total_deletions / total_refs,
                "insertions": total_insertions / total_refs,
                "hits": total_hits / total_refs,
            },
        }
    
