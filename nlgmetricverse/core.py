"""
Main application file. Contains Nlgmetricverse main class.
"""
import warnings
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from nlgmetricverse.collator import Collator
from nlgmetricverse.data_loader import DataLoader
from nlgmetricverse.definitions import DEFAULT_METRICS
from nlgmetricverse.metrics import EvaluationInstance, Metric, load_metric
from nlgmetricverse.utils.common import pop_item_from_dict, replace, set_env
from nlgmetricverse import data_loader

MetricParam = Union[str, Metric, Dict[str, Any]]


class Nlgmetricverse:
    r"""
    Simple evaluation pipeline for text based metrics. By default, it computes BLEU(n), METEOR, ROUGE-L metrics.
    """

    def __init__(
            self,
            metrics: Optional[Union[MetricParam, List[MetricParam]]] = None,
            run_concurrent=False,
    ):
        self.metrics = self._load_metrics(metrics)
        self._concurrent = run_concurrent
        self.res_predictions: EvaluationInstance = None,
        self.res_references: EvaluationInstance = None,

        # Sanity check
        self._validate_metrics()

    def __call__(
            self,
            *,
            predictions: str,
            references: str,
            reduce_fn: Optional[Union[str, Callable]] = None,
            method: str = "no_new_line",
            **kwargs
    ) -> Dict[str, float]:
        """
        Restricts positional arguments to prevent potential inconsistency between predictions and references.

        :param predictions: Predictions.
        :param references: References.
        :param reduce_fn: Reduce function name.
        :param method: Method to analyse inputs. Can be "no_new_line" or "read_lines"
        :return: scores
        """

        dl = data_loader.DataLoader(predictions, references, method)
        self.res_predictions = dl.get_predictions()
        self.res_references = dl.get_references()
        scores = dict()
        scores["total_items"] = len(self.res_references)
        scores["empty_items"] = self._remove_empty(self.res_predictions, self.res_references)

        if scores["total_items"] == scores["empty_items"]:
            warnings.warn(
                "At least one of the pairs are empty for all evaluation instances. No evaluation takes place."
            )
            return scores

        if self._concurrent:
            inputs_list = self._prepare_concurrent_inputs(self.res_predictions, self.res_references, reduce_fn, kwargs)
            set_env("TOKENIZERS_PARALLELISM", "true")
            with ProcessPoolExecutor() as executor:
                for score in executor.map(self._compute_single_score, inputs_list):
                    scores.update(score)
        else:
            for metric in self.metrics:
                inputs = (metric, self.res_predictions, self.res_references, reduce_fn, kwargs)
                score = self._compute_single_score(inputs)
                scores.update(score)

        return scores

    @staticmethod
    def _remove_empty(predictions: EvaluationInstance, references: EvaluationInstance):
        """
        Remove empty predictions and references.

        :param predictions: Predictions.
        :param references: References.
        :return: Number of empty elements.
        """
        n_items = len(predictions)
        n_empty = 0
        for i in reversed(range(n_items)):
            if not isinstance(predictions[i], (float, int)) and not isinstance(references[i], (float, int)):
                if not predictions[i] or not references[i]:
                    predictions.pop(i)
                    references.pop(i)
                    n_empty += 1
        return n_empty

    @staticmethod
    def _load_single_metric(metric: Union[str, Metric]) -> List[Metric]:
        """
        Load a single metric.

        :param metric: Metric to be loaded.
        :return: Loaded metric.
        """
        if isinstance(metric, str):
            metric = load_metric(metric)
        return [metric]

    @staticmethod
    def _load_multiple_metrics(metrics: Union[List[str], List[Dict[str, Any]], List[Metric]]) -> List[Metric]:
        """
        Load multiple metrics.

        :param metrics: Metrics to be loaded.
        :return: Loaded metrics.
        """
        for i, metric_param in enumerate(metrics):
            if isinstance(metric_param, str):
                path = metric_param
                metrics = replace(metrics, load_metric(path.lower()), i)
            elif isinstance(metric_param, dict):
                path = metric_param.pop("path")  # must be given
                task = pop_item_from_dict(metric_param, "task")
                resulting_name = pop_item_from_dict(metric_param, "resulting_name")
                compute_kwargs = pop_item_from_dict(metric_param, "compute_kwargs")
                kwargs = metric_param
                metrics = replace(
                    metrics,
                    load_metric(
                        path=path,
                        task=task,
                        resulting_name=resulting_name,
                        compute_kwargs=compute_kwargs,
                        **kwargs,
                    ),
                    i,
                )
            elif isinstance(metric_param, Metric):
                continue
        return metrics

    def _load_metrics(self, metrics: Union[MetricParam, List[MetricParam]]) -> List[Metric]:
        """
        Load all metrics.

        :param metrics: All metrics to be loaded.
        :return: All metrics loaded.
        """
        if metrics is None:
            metrics = DEFAULT_METRICS

        if isinstance(metrics, (str, Metric)):
            metrics = self._load_single_metric(metrics)
        elif isinstance(metrics, list):
            metrics = self._load_multiple_metrics(metrics)
        else:
            raise ValueError(f"Unknown input type {type(metrics)}")

        return metrics

    @staticmethod
    def _score_to_dict(score, name: str) -> Dict[str, float]:
        """
        Transform score to dict.

        :param score: score
        :param name: Name of the score.
        :return: Dict containing score and name.
        """
        if isinstance(score, dict):
            return score

        return {name: score}

    def _compute_single_score(self, inputs) -> Mapping[str, float]:
        """
        Compute a single score.

        :param inputs: Inputs parameters.
        :return: score
        """
        metric, predictions, references, reduce_fn, kwargs = inputs
        start = time.time()
        if isinstance(metric, Metric):
            predictions, references = Collator(predictions), Collator(references)
            score = metric.compute(predictions=predictions, references=references, reduce_fn=reduce_fn, **kwargs)
        else:
            metric.resulting_name = metric.name
            score = metric.compute(predictions=predictions, references=references, **kwargs)
            score = self._score_to_dict(score, name=metric.name)
        end = time.time()
        print("time elapsed computing " + metric.resulting_name + ": " + str(end - start) + " sec")
        return score

    def _prepare_concurrent_inputs(self, predictions, references, reduce_fn, kwargs):
        """
        Prepare concurrent computing.

        :param predictions: Predictions.
        :param references: References.
        :param reduce_fn: Reduce function name.
        :return: Vect of inputs.
        """
        inputs = []
        for metric in self.metrics:
            inputs.append((metric, predictions, references, reduce_fn, kwargs))
        return inputs

    def _validate_metrics(self):
        """
        Validate if metrics are of the same task.

        :return: True if metrics are of the same task.
        """
        metrics = self.metrics
        if all([isinstance(metric, Metric) for metric in metrics]):
            task = metrics[0].task
            if not all([metric.task == task for metric in metrics]):
                raise ValueError(
                    "Given metrics are not suitable to be used together, metrics must be of same the task."
                )
        return True

    def add_metric(self, path: str, resulting_name: str = None, compute_kwargs: Dict = None) -> None:
        """
        Add new metric to computing.

        :param path: Metric path.
        :param resulting_name: Metric name.
        :param compute_kwargs: Additional arguments for correct output.
        :return: None.
        """
        metric = load_metric(path, resulting_name=resulting_name, compute_kwargs=compute_kwargs)
        if metric not in self.metrics:
            self.metrics.append(metric)
            self._validate_metrics()

    def remove_metric(self, resulting_name: str, error: bool = True) -> None:
        """
        Remove metric from computing.

        :param resulting_name: Metric name.
        :param error: Raise an error if resulting_metric is not found
        :return: None.
        """
        for i, metric in enumerate(self.metrics):
            if metric.resulting_name == resulting_name:
                self.metrics.pop(i)
                return
        if error:
            raise ValueError(f"Metric with resulting name {resulting_name} does not exists.")

    def evaluate(
            self,
            *,
            predictions: Union[List[str], List[List[str]]] = None,
            references: Union[List[str], List[List[str]]] = None,
            reduce_fn: Optional[Union[str, Callable]] = None,
    ) -> Dict[str, float]:
        """
        Returns __call__() method. For backward compatibility.

        :param predictions: Predictions.
        :param references: References.
        :param reduce_fn: Reduce function name.
        :return: ___call__() method.
        """
        return self.__call__(predictions=predictions, references=references, reduce_fn=reduce_fn)
