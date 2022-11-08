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
Comet cross-lingual MT evaluation metric. The part of this file is adapted from
Comet implementation of evaluate package. See
https://github.com/huggingface/evaluate/blob/master/metrics/comet/comet.py
"""

from typing import Callable, Union

import evaluate

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.metrics._core import MetricForLanguageGeneration
from nlgmetricverse.metrics._core.utils import PackagePlaceholder, requirement_message

# `import comet` placeholder
comet = PackagePlaceholder(version="1.0.1")

_CITATION = """\
@inproceedings{rei-EtAl:2020:WMT,
   author    = {Rei, Ricardo  and  Stewart, Craig  and  Farinha, Ana C  and  Lavie, Alon},
   title     = {Unbabel's Participation in the WMT20 Metrics Shared Task},
   booktitle      = {Proceedings of the Fifth Conference on Machine Translation},
   month          = {November},
   year           = {2020},
   address        = {Online},
   publisher      = {Association for Computational Linguistics},
   pages     = {909--918},
}
@inproceedings{rei-etal-2020-comet,
   title = "{COMET}: A Neural Framework for {MT} Evaluation",
   author = "Rei, Ricardo  and
      Stewart, Craig  and
      Farinha, Ana C  and
      Lavie, Alon",
   booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
   month = nov,
   year = "2020",
   address = "Online",
   publisher = "Association for Computational Linguistics",
   url = "https://www.aclweb.org/anthology/2020.emnlp-main.213",
   pages = "2685--2702",
}
"""

_DESCRIPTION = """\
Crosslingual Optimized Metric for Evaluation of Translation (COMET) is an open-source neural framework for
generating multilingual-MT evaluation prediction estimates of three types of human judgments (HTER, DA's
or MQM), training a model for each judgment type, achieving high-level correlations with the ground-truth
scores and better robustness. To encompass the distinct scoring types, the COMET framework supports two
architectures with differnet training objectives: (i) the Estimator model (targets = real values, i.e.,
HTER and MQM); (ii) the Translation Ranking model (targets = relative rankings, i.e., DA). While the
Estimator is trained to regress directly on a quality score, the Translation Ranking model is trained to
minimize the distance between a "better" hypothesis and both its corresponding reference and its original
source. Both models are composed of a pre-trained cross-lingual encoder (e.g., XLM-RoBERTa, multilingual
BERT), and a pooling layer to produce sentence embeddings.
- The Estimator model independently encode the hypothesis and the reference (encoding), transforming the
  word embeddings into a sentence embedding for each segment (pooling). Finally, the resulting sentence
  embeddings are combined and concatenated into one single vector that is passed to a feed-forward
  regressor. The entire model is trained by minimizing the Mean Squared Error (MSE).
- The Translation Ranking model receives 4 segments: the source, the reference, a "better" hypothesis,
  and a "worse" one. These segments are independently encoded using a pretrained cross-lingual encoder
  and a pooling layer on top. Finally, using the triplet margin loss (Schroff et al., 2015), the resulting
  embedding space is optimized to minimize the distance between the "better" hypothesis and the "anchors"
  (source and reference).

With the release of the framework the authors also released fully trained models that were used to
compete in the WMT20 Metrics Shared Task achieving SOTA in that years competition.
"""

_KWARGS_DESCRIPTION = """
COMET score.
Args:
`sources` (list of str): source sentences.
`predictions` (list of str): candidate translations.
`references` (list of str): reference translations.
`gpus` (int): optional, an integer (number of GPUs to train on) or a list of integers (which GPUs to train on). Set to 0 to use CPU. The default value is None (uses one GPU if possible, else use CPU).
`progress_bar` (bool): if set to True, progress updates will be printed out. The default value is False.
`config_name` (str): COMET model to be used. Will default to wmt20-comet-da (previously known as wmt-large-da-estimator-1719) if None.
    Alternate models that can be chosen include wmt20-comet-qe-da, wmt21-comet-mqm, wmt21-cometinho-da, wmt21-comet-qe-mqm and emnlp20-comet-rank.
Returns:
    `samples` (float): the mean value of COMET `scores` over all the input sentences.
    `scores` (list of floats): List of scores.
Examples:
    >>> scorer = NLGMetricverse(metrics=load_metric("comet"))
    >>> sources = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
    >>> predictions = ["The fire could be stopped", "Schools and kindergartens were open"]
    >>> references = ["They were able to control the fire", "Schools and kindergartens opened"]
    >>> scores = scorer(sources=sources, predictions=predictions, references=references)
    >>> print(scores)
    { "total_items": 2, "empty_items": 0, "comet": { "scores": [ 0.1506408303976059, 0.915494441986084 ], "samples": 0.5330676361918449 } }
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CometPlanet(MetricForLanguageGeneration):
    def _download_and_prepare(self, dl_manager):
        global comet
        try:
            import comet
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="comet", package_name="unbabel-comet"))
        else:
            super(CometPlanet, self)._download_and_prepare(dl_manager)

        if self.config_name == "default":
            checkpoint_path = comet.download_model("wmt20-comet-da")
        else:
            checkpoint_path = comet.download_model(self.config_name)
        self.scorer = comet.load_from_checkpoint(checkpoint_path)

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://unbabel.github.io/COMET/html/index.html",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/Unbabel/COMET"],
            reference_urls=[
                "https://github.com/Unbabel/COMET",
                "https://www.aclweb.org/anthology/2020.emnlp-main.213/",
                "http://www.statmt.org/wmt20/pdf/2020.wmt-1.101.pdf6",
            ],
        )

    def _compute_single_pred_single_ref(
        self,
        sources: EvaluationInstance,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        reduce_fn: Callable = None,
        batch_size: int = 8,
        gpus: int = 1,
        mc_dropout: Union[int, bool] = False,
        progress_bar: bool = True,
        accelerator: str = "ddp",
        num_workers: int = None,
        length_batching: bool = True,
    ):
        data = {"src": sources, "mt": predictions, "ref": references}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        scores, samples = self.scorer.predict(
            data,
            batch_size=batch_size,
            gpus=gpus,
            mc_dropout=mc_dropout,
            progress_bar=progress_bar,
            accelerator=accelerator,
            num_workers=num_workers,
            length_batching=length_batching,
        )
        return {"scores": scores, "samples": samples}

    def _compute_single_pred_multi_ref(
        self,
        sources: EvaluationInstance,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        reduce_fn: Callable = None,
        batch_size: int = 8,
        gpus: int = 1,
        mc_dropout: Union[int, bool] = False,
        progress_bar: bool = True,
        accelerator: str = "ddp",
        num_workers: int = None,
        length_batching: bool = True,
    ):
        scores = []
        for src, pred, refs in zip(sources, predictions, references):
            data = {"src": [src] * len(refs), "mt": [pred] * len(refs), "ref": refs}
            data = [dict(zip(data, t)) for t in zip(*data.values())]
            pred_scores, _ = self.scorer.predict(
                data,
                batch_size=batch_size,
                gpus=gpus,
                mc_dropout=mc_dropout,
                progress_bar=progress_bar,
                accelerator=accelerator,
                num_workers=num_workers,
                length_batching=length_batching,
            )
            scores.append(float(reduce_fn(pred_scores)))

        return {"scores": scores, "samples": sum(scores) / len(scores)}

    def _compute_multi_pred_multi_ref(
        self,
        sources: EvaluationInstance,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        reduce_fn: Callable = None,
        batch_size: int = 8,
        gpus: int = 1,
        mc_dropout: Union[int, bool] = False,
        progress_bar: bool = True,
        accelerator: str = "ddp",
        num_workers: int = None,
        length_batching: bool = True,
    ):
        scores = []
        for src, preds, refs in zip(sources, predictions, references):
            all_pred_scores = []
            for pred in preds:
                data = {"src": [src] * len(refs), "mt": [pred] * len(refs), "ref": refs}
                data = [dict(zip(data, t)) for t in zip(*data.values())]
                pred_scores, _ = self.scorer.predict(
                    data,
                    batch_size=batch_size,
                    gpus=gpus,
                    mc_dropout=mc_dropout,
                    progress_bar=progress_bar,
                    accelerator=accelerator,
                    num_workers=num_workers,
                    length_batching=length_batching,
                )
                all_pred_scores.append(float(reduce_fn(pred_scores)))
            scores.append(float(reduce_fn(all_pred_scores)))

        return {"scores": scores, "samples": sum(scores) / len(scores)}
