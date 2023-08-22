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
BLEURT metric. The part of this file is adapted from BLEURT implementation
of evaluate package. See
https://github.com/huggingface/evaluate/blob/master/metrics/bleurt/bleurt.py
"""

import os
from typing import Callable

import evaluate

from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics import EvaluationInstance, MetricForLanguageGeneration
from nlgmetricverse.metrics._core.utils import PackagePlaceholder, requirement_message
from nlgmetricverse.utils.sys import log

# `import bleurt` placeholder
bleurt = PackagePlaceholder(version="1.2.2")

logger = evaluate.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{bleurt,
  title={BLEURT: Learning Robust Metrics for Text Generation},
  author={Thibault Sellam and Dipanjan Das and Ankur P. Parikh},
  booktitle={ACL},
  year={2020},
  url={https://arxiv.org/abs/2004.04696}
}
"""

_DESCRIPTION = """\
Bilingual Evaluation Understudy with Representations from Transformers (BLEURT) is a fully learned evaluation
metric modeling human judgments for generated text. BLEURT is based on BERT (Devlin et al. 2018) and a novel
(additional) pre-training scheme based on millions of synthetic reference-candidate pairs, generated through
perturbations (i.e., mask-filling, backtranslation, dropping words) and aimed to help the model generalize
(greater robustness). Differently from existing sentence pairs evaluate, synthetic data allow to capture the
errors and alterations that NLG systems produce (e.g., omissions, repetitions, nonsensical substitutions).
Extra BERT pre-training on such syntethic data considers several lexical- and semantic-level supervision
signals with a multitask loss, i.e., a weighted sum aggregation of task-level regression or classification
losses (BLEU/ROUGE/BERTScore emulation, backtranslation likelihood/flag, textual entailment).
So, BLEURT models are trained in three steps: regular BERT pre-training (Devlin et al., 2019), pre-training
on synthetic data, and fine-tuning on task-specific ratings (like translation and/or data-to-text using
public WMT human annotations). Note: rating data prediction at the third step is done with a classification
layer on top of BERT's [CLS].

You may run BLEURT  out-of-the-box or fine-tune it for your specific application (the latter is expected to
perform better). See the project's README at https://github.com/google-research/bleurt#readme for more
information.
"""

_KWARGS_DESCRIPTION = """
BLEURT score.
Args:
    `predictions` (list of str): prediction/candidate sentences
    `references` (list of str): reference sentences
    `config_name` (str): BLEURT checkpoint. Will default to BLEURT-base if None.
Returns:
    'score': Average BLEURT score.
    'scores': List of BLEURT scores.
    'checkpoint': Selected BLEURT checkpoint.
Examples:
    >>> scorer = NLGMetricverse(metrics=load_metric("bertscore", config_name="bleurt-tiny-128"))
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
    {'total_items': 2, 'empty_items': 0, 'bleurt': {'score': 0.6418270468711853,
    'scores': [0.47344332933425903, 0.8102107644081116], 'checkpoint': 'bleurt-tiny-128'}}
"""

_LICENSE = """Copyright 2021 Google.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

CHECKPOINT_URLS = {
    "bleurt-tiny-128": "https://storage.googleapis.com/bleurt-oss/bleurt-tiny-128.zip",
    "bleurt-tiny-512": "https://storage.googleapis.com/bleurt-oss/bleurt-tiny-512.zip",
    "bleurt-base-128": "https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip",
    "bleurt-base-512": "https://storage.googleapis.com/bleurt-oss/bleurt-base-512.zip",
    "bleurt-large-128": "https://storage.googleapis.com/bleurt-oss/bleurt-large-128.zip",
    "bleurt-large-512": "https://storage.googleapis.com/bleurt-oss/bleurt-large-512.zip",
    "BLEURT-20-D3": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D3.zip",
    "BLEURT-20-D6": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D6.zip",
    "BLEURT-20-D12": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip",
    "BLEURT-20": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip",
}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BleurtPlanet(MetricForLanguageGeneration):
    def _info(self):
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/google-research/bleurt",
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=1,
            lower_bound=0,
            features=self._default_features,
            codebase_urls=["https://github.com/google-research/bleurt"],
            reference_urls=["https://github.com/google-research/bleurt", "https://arxiv.org/abs/2004.04696"],
            license=_LICENSE,
        )

    def _download_and_prepare(self, dl_manager):
        global bleurt
        try:
            from bleurt import score
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                requirement_message(
                    path="bleurt",
                    package_name="bleurt"
                )
            )

        # check that config name specifies a valid BLEURT model
        if self.config_name == "default":
            logger.warning(
                "Using default BLEURT-Base checkpoint for sequence maximum length 128. "
                "You can use a bigger model for better results with e.g: "
                "NLGMetricverse(metric={path: 'bleurt', 'checkpoint': 'bleurt-large-512'})."
            )
            self.config_name = "bleurt-base-128"

        if self.config_name.lower() in CHECKPOINT_URLS:
            checkpoint_name = self.config_name.lower()

        elif self.config_name.upper() in CHECKPOINT_URLS:
            checkpoint_name = self.config_name.upper()

        else:
            raise KeyError(
                f"{self.config_name} model not found. You should supply the name of a model checkpoint for bleurt in "
                f"{CHECKPOINT_URLS.keys()}"
            )

        # download the model checkpoint specified by self.config_name and set up the scorer
        model_path = dl_manager.download_and_extract(CHECKPOINT_URLS[checkpoint_name])
        self.scorer = score.BleurtScorer(os.path.join(model_path, checkpoint_name))

    def _compute_single_pred_single_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        """
        Compute the bleurt score for a single prediction and a single reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a single text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
        """
        scores = self.scorer.score(references=references, candidates=predictions)
        return {"score": sum(scores) / len(scores), "scores": scores, "checkpoint": self.config_name}

    def _compute_single_pred_multi_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        """
        Compute the bleurt score for a single prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
        """
        scores = []
        for pred, refs in zip(predictions, references):
            pred = [pred] * len(refs)
            pred_scores = self.scorer.score(references=refs, candidates=pred)
            reduced_score = float(reduce_fn(pred_scores))
            scores.append(reduced_score)
        return {"score": sum(scores) / len(scores), "scores": scores, "checkpoint": self.config_name}

    def _compute_multi_pred_multi_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        """
        Compute the bleurt score for multiple prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a multiple text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
        """
        scores = []
        for preds, refs in zip(predictions, references):
            pred_scores = []
            for pred in preds:
                pred = [pred] * len(refs)
                pred_score = self.scorer.score(references=refs, candidates=pred)
                pred_scores.append(reduce_fn(pred_score))
            reduced_score = float(reduce_fn(pred_scores))
            scores.append(reduced_score)

        return {"score": sum(scores) / len(scores), "scores": scores, "checkpoint": self.config_name}


if __name__ == "__main__":
    import json

    predictions = [["hello there", "general kenobi"]]
    references = [["hello there", "general kenobi"]]
    bleurt = BleurtPlanet()
    results = bleurt.compute(predictions=predictions, references=references)
    log(json.dumps(results, indent=2))
