"""
Many utils functions for metrics.
"""

import importlib.util
import os
import warnings
from pathlib import Path
from typing import Callable, Sequence, Union
import numpy as np
import requests
import json
from enum import Enum

PACKAGE_CORE = Path(os.path.abspath(os.path.dirname(__file__)))
METRICS_ROOT = PACKAGE_CORE.parent
PACKAGE_SOURCE = METRICS_ROOT.parent
PROJECT_ROOT = PACKAGE_SOURCE.parent


class PackagePlaceholder:
    def __init__(self, version: str):
        self.__version__ = version


class TaskNotAvailable(KeyError):
    def __init__(self, path: str, task: str):
        message = f"Task '{task}' is not available for metric '{path}'."
        self.message = message
        super(TaskNotAvailable, self).__init__(message)


def requirement_message(path: str, package_name: str) -> str:
    """
    Show a message listing the required packages for a specific metric.

    :param path: Metric path.
    :param package_name: Package name.
    :return: A message.
    """
    return (
        f"In order to use metric '{path}', '{package_name}' is required. "
        f"You can install the package by `pip install {package_name}`."
    )


def download(source: str, destination: str, overwrite: bool = False, warn: bool = False) -> None:
    """
    Download a package.

    :param source: Package sources.
    :param destination: Where to install it.
    :param overwrite: Bool.
    :param warn: Bool.
    :return: None.
    """
    if os.path.exists(destination) and not overwrite:
        if warn:
            warnings.warn(
                f"Path {destination} already exists, not overwriting. To overwrite, specify " f"'overwrite' parameter."
            )
        return
    r = requests.get(source, allow_redirects=True)

    with open(destination, "wb") as out_file:
        out_file.write(r.content)


def import_module(module_name: str, filepath: str):
    """
    Import a module.

    :param module_name: Module name.
    :param filepath: File path.
    :return: Module.
    """
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_token_lengths(sequences: Sequence[Sequence[str]], reduce_fn: Callable = None) -> Union[int, Sequence[int]]:
    """
    Get token lengths.

    :param sequences: Sequencs of tokens.
    :param reduce_fn: Reduce function name.
    :return: Token lengths.
    """
    token_lengths = [len(item) for item in sequences]
    if reduce_fn is not None:
        return int(reduce_fn(token_lengths))
    return token_lengths


def is_reduce_fn(fun: Callable) -> bool:
    """
    Check if a function is a reduce function.

    :param fun: Function name.
    :return: True if function is a reduce function, else 0.
    """
    result = np.array(fun([1, 2]))
    return result.size == 1


def list_metrics():
    """
    Get all metrics installed.

    :return: All metrics installed.
    """
    _internal_metrics_path = METRICS_ROOT
    metric_modules = list(_internal_metrics_path.glob("[!_]*"))
    return [module_name.name.replace(".py", "") for module_name in metric_modules]


class Categories(Enum):
    Embedding = "embedding-based"
    Overlap = "n-gram overlap"
    Distance = "distance-based"


class ApplTasks(Enum):
    DataToText = "D2T"
    MachineTranslation = "MT"
    DocumentSummarization = "SUM"
    ImageCaptioning = "IC"
    SpeechRecognition = "SR"
    DocumentGeneration = "DG"
    QuestionGeneration = "QG"
    ResponseGeneration = "RG"


class QualityDims(Enum):
    Informativeness = "INFO"
    Relevance = "REL"
    Fluency = "FLU"
    Coherence = "COH"
    Factuality = "FAC"
    SemanticCoverage = "COV"
    Adequacy = "ADE"


def __apply_category_filter(metrics, category: Categories = None):
    res = []
    if category is None:
        res = metrics
    else:
        for metric in metrics:
            if metric["category"] == category.value:
                res.append(metric)
    return res


def __apply_task_filter(metrics, appl_task: ApplTasks = None):
    res = []
    if appl_task is None:
        res = metrics
    else:
        for metric in metrics:
            if appl_task.value in metric["appl_tasks"]:
                res.append(metric)
    return res


def __apply_trained_filter(metrics, trained: bool = None):
    res = []
    if trained is None:
        res = metrics
    else:
        for metric in metrics:
            if metric["trained"] == trained:
                res.append(metric)
    return res


def __apply_unsupervised_filter(metrics, unsupervised: bool = None):
    res = []
    if unsupervised is None:
        res = metrics
    else:
        for metric in metrics:
            if metric["unsupervised"] == unsupervised:
                res.append(metric)
    return res


def __apply_quality_filter(metrics, quality_dim: ApplTasks = None):
    res = []
    if quality_dim is None:
        res = metrics
    else:
        for metric in metrics:
            if quality_dim.value in metric["quality_dims"]:
                res.append(metric)
    return res


def filter_metrics(category: Categories = None, appl_task: ApplTasks = None,
                   trained: bool = None, unsupervised: bool = None, quality_dim: QualityDims = None):
    os.chdir("nlgmetricverse/metrics")
    f = open('list_metrics.json')
    data = json.load(f)
    metrics = __apply_category_filter(data['metrics'], category)
    metrics = __apply_task_filter(metrics, appl_task)
    metrics = __apply_trained_filter(metrics, trained)
    metrics = __apply_unsupervised_filter(metrics, unsupervised)
    metrics = __apply_quality_filter(metrics, quality_dim)
    res = []
    for metric in metrics:
        res.append(metric["name"])
    f.close()
    return res
