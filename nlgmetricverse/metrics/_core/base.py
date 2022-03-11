# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors
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
Metrics base class. The part of this file is adapted from HuggingFace's
datasets package implementation of Accuracy metric. See
https://github.com/huggingface/datasets/blob/master/src/datasets/metric.py
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import numpy
import pandas as pd
from datasets.utils.logging import get_logger

from nlgmetricverse.collator import Collator
from nlgmetricverse.metrics._core.utils import import_module, is_reduce_fn

LanguageGenerationInstance = Union[List[str], List[List[str]]]
SequenceClassificationInstance = Union[List[int], List[List[int]]]
SequenceLabelingInstance = List[List[str]]
EvaluationInstance = Union[LanguageGenerationInstance, SequenceClassificationInstance, SequenceLabelingInstance]
MetricOutput = Dict[str, Union[str, int, float]]

logger = get_logger(__name__)


class Metric(datasets.Metric, ABC):
    """
    Base metric class and common API for all metrics.
    """

    def __init__(
        self,
        task: str,
        resulting_name: Optional[str] = None,
        compute_kwargs: Optional[Dict[str, Any]] = None,
        config_name: Optional[str] = None,
        keep_in_memory: bool = False,
        cache_dir: Optional[str] = None,
        num_process: int = 1,
        process_id: int = 0,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
        max_concurrent_cache_files: int = 10000,
        timeout: Union[int, float] = 100,
        **kwargs,
    ):
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=cache_dir,
            num_process=num_process,
            process_id=process_id,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=max_concurrent_cache_files,
            timeout=timeout,
            **kwargs,
        )
        self._task = task
        self.resulting_name = resulting_name if resulting_name is not None else self.name
        self.compute_kwargs = compute_kwargs or {}
        self.download_and_prepare()

    @abstractmethod
    def _compute(
        self,
        *,
        predictions: EvaluationInstance = None,
        references: EvaluationInstance = None,
        **kwargs,
    ) -> MetricOutput:
        """
        Base compute method which is used for internal computation. All child classes must implement _compute() method.
        """
        pass

    @abstractmethod
    def _compute_single_pred_single_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and single reference case.

        :param predictions: Predictions.
        :param references: References.
        :param kwargs: Additional arguments used for the metric computation.
        :return: score
        """
        pass

    @abstractmethod
    def _compute_single_pred_multi_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and multiple references case.

        :param predictions: Predictions.
        :param references: References.
        :param kwargs: Additional arguments used for the metric computation.
        :return: score
        """
        pass

    @abstractmethod
    def _compute_multi_pred_multi_ref(
        self,
        predictions: EvaluationInstance,
        references: EvaluationInstance,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and multiple references case.

        :param predictions: Predictions.
        :param references: References.
        :param kwargs: Additional arguments used for the metric computation.
        :return: score
        """
        pass

    def _download_and_prepare(self, dl_manager):
        """
        Downloads and prepares resources for the metric. This is the internal implementation to overwrite called when
        user calls `download_and_prepare`. It should download all required resources for the metric.

        :param dl_manager: `DownloadManager` used to download and cache data.
        :return: None
        """
        self.external_module_path = None
        return None

    def _get_external_resource(self, module_name: Optional[str], attr: Optional[str] = None):
        """
        Get external resources for the metric.

        :param module_name: Optional module to get.
        :param attr: Optional attribute for the resource.
        :return: External module attribute.
        """
        if self.external_module_path is None:
            raise AttributeError("'external_module_path' is not defined.")
        if module_name is None:
            module_name = "external_module"
        external_module = import_module(module_name, self.external_module_path)
        if attr is None:
            return external_module
        return getattr(external_module, attr)

    @property
    def task(self):
        return self._task


class MetricForTask(Metric, ABC):
    """
    Base metric class for any task. All metrics must extend this class as metric is required to adopt a task inherently.
    Default task will be language-generation for AutoMetric.
    All metrics extending :py:class:`nlgmetricverse.metrics._core.base.MetricForTask` must implement the following:
        - _task (``[str]``): Task name for the base task metric.
        - _default_features() (``datasets.Features``): Task input as a :py:class:`datasets.Features`.
    """

    _task = None

    def __init__(self, resulting_name: Optional[str] = None, compute_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        compute_kwargs = self._validate_compute_kwargs(compute_kwargs)
        super().__init__(task=self._task, resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _validate_compute_kwargs(self, compute_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the computed kwargs.

        :param compute_kwargs: Additional arguments for correct output.
        :return: Computed additional arguments.
        """
        if compute_kwargs is not None and "reduce_fn" in compute_kwargs:
            compute_kwargs.pop("reduce_fn")
        return compute_kwargs

    @property
    def _default_features(self):
        raise NotImplementedError

    @classmethod
    def _construct(
        cls, resulting_name: Optional[str] = None, compute_kwargs: Optional[Dict[str, Any]] = None, **kwargs
    ):
        return cls(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    @staticmethod
    def _reduce_scores(scores: Union[List[Dict[str, float]], List[float]], reduce_fn: Callable):
        """
        Reduce the scores by a given function.

        :param scores: Scores to be reduced.
        :param reduce_fn: Reduce function.
        :return: score
        """
        if isinstance(scores[0], dict):
            score = pd.DataFrame(scores).apply(reduce_fn, axis=0).to_dict()
        else:
            score = float(reduce_fn(scores))
        return score

    @staticmethod
    def _preprocess(predictions: List[List[str]], references: List[List[str]]) -> Tuple[Collator, Collator]:
        """
        Preprocess predictions and references.

        :param predictions: Predictions.
        :param references: References.
        :return: Collator class for predictions and references.
        """
        return Collator(predictions, keep=True), Collator(references, keep=True)

    def _compute(
        self,
        *,
        predictions: EvaluationInstance = None,
        references: EvaluationInstance = None,
        **kwargs,
    ) -> MetricOutput:
        """
        Compute the result for predictions from references by the given metric.

        :param predictions: Predictions.
        :param references: References.
        :param kwargs: Additional arguments used for the metric computation.
        :return: result
        """
        assert len(predictions) == len(references), "Predictions and references length does not match."
        eval_params = {**self.compute_kwargs, **kwargs}
        if "reduce_fn" in eval_params:
            eval_params.pop("reduce_fn")
        predictions, references = Collator(predictions), Collator(references)
        result = self.evaluate(predictions=predictions, references=references, **eval_params)
        return {self.resulting_name: result}

    def evaluate(self, predictions: Collator, references: Collator, **kwargs) -> Dict[str, float]:
        """
        Evaluate the results.

        :param predictions: Predictions.
        :param references: References.
        :param kwargs: Additional arguments used for the metric computation.
        :return: score
        """
        predictions, references = self._preprocess(predictions, references)
        if predictions.can_collapse() and references.can_collapse():
            predictions = predictions.collapse()
            references = references.collapse()
            eval_fn = self._compute_single_pred_single_ref
        elif predictions.can_collapse() and not references.can_collapse():
            predictions = predictions.collapse()
            eval_fn = self._compute_single_pred_multi_ref
        else:
            eval_fn = self._compute_multi_pred_multi_ref
        return eval_fn(predictions=predictions, references=references, **kwargs)


class MetricForLanguageGeneration(MetricForTask):
    """
    Base metric class for language generation task. All metrics on nlgmetricverse are language generation metrics
    which are used by default by :py:class:`nlgmetricverse.metrics.AutoMetric`.
    """

    _task = "language-generation"

    @property
    def _default_features(self):
        return datasets.Features(
            {
                "predictions": datasets.Sequence(datasets.Value("string", id="sequence")),
                "references": datasets.Sequence(datasets.Value("string", id="sequence")),
            }
        )

    def _validate_compute_kwargs(self, compute_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the computed kwargs.

        :param compute_kwargs: Additional arguments for correct output.
        :return: Computed additional arguments.
        """
        if compute_kwargs is None:
            compute_kwargs = {}
        if "reduce_fn" not in compute_kwargs:
            compute_kwargs.update({"reduce_fn": "max"})
        return compute_kwargs

    def _compute(
        self,
        *,
        predictions: LanguageGenerationInstance = None,
        references: LanguageGenerationInstance = None,
        **kwargs,
    ) -> MetricOutput:
        """
        Compute the result for predictions from references by the given metric.

        :param predictions: Predictions.
        :param references: References.
        :param kwargs: Additional arguments used for the metric computation.
        :return: result
        """
        assert len(predictions) == len(references), "Predictions and references length does not match."
        reduce_fn = kwargs.get("reduce_fn")
        reduce_fn = self.compute_kwargs["reduce_fn"] if reduce_fn is None else reduce_fn
        if isinstance(reduce_fn, str):
            reduce_fn = getattr(numpy, reduce_fn)
        elif reduce_fn is not None and not callable(reduce_fn):
            raise TypeError(f"'reduce_fn' Expected str or callable, got {type(reduce_fn)}")
        if reduce_fn is not None and not is_reduce_fn(reduce_fn):
            raise ValueError("'reduce_fn' must be an aggregation function.")
        eval_params = {**self.compute_kwargs, **kwargs}
        eval_params.pop("reduce_fn")
        predictions, references = Collator(predictions), Collator(references)
        result = self.evaluate(predictions=predictions, references=references, reduce_fn=reduce_fn, **eval_params)
        return {self.resulting_name: result}

    @abstractmethod
    def _compute_single_pred_single_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and single reference case.

        :param predictions: Predictions.
        :param references: References.
        :param reduce_fn: Reduce function name.
        :param kwargs: Additional arguments used for the metric computation.
        :return: score
        """
        pass

    @abstractmethod
    def _compute_single_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        """
        Computes the metric score(s) for single prediction and multiple references case.

        :param predictions: Predictions.
        :param references: References.
        :param reduce_fn: Reduce function name.
        :param kwargs: Additional arguments used for the metric computation.
        :return: score
        """
        pass

    @abstractmethod
    def _compute_multi_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        """
        Computes the metric score(s) for multiple prediction and multiple references case.

        :param predictions: Predictions.
        :param references: References.
        :param reduce_fn: Reduce function name.
        :param kwargs: Additional arguments used for the metric computation.
        :return: score
        """
        pass
