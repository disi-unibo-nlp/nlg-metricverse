import inspect
import json
import os
from typing import Optional

import pytest

from nlgmetricverse import NLGMetricverse, load_metric
from nlgmetricverse.utils.sys import set_env
from tests.nlgmetricverse import EXPECTED_OUTPUTS

# Setting cuda visible devices to None
# assessing CPU only computation for test suite.
set_env("CUDA_VISIBLE_DEVICES", "-1")

_TEST_METRICS = [
    load_metric("bleu"),
    load_metric("meteor"),
    load_metric("rouge"),
    load_metric("sacrebleu")
]

_STR_TEST_METRIC = "bleu"

_LIST_STR_TEST_METRICS = [
    "bleu",
    "meteor",
    "rouge",
    "sacrebleu"
]

_LIST_DICT_TEST_METRICS = [
    {"path": "bleu", "resulting_name": "bleu-1", "compute_kwargs": {"max_order": 1}},
    {"path": "bleu", "resulting_name": "bleu-2", "compute_kwargs": {"max_order": 2}},
    {"path": "meteor", "resulting_name": "METEOR"},
    {"path": "rouge"},
    {"path": "sacrebleu"}
]

_LIST_MIXED_TEST_METRICS = [
    "bleu",
    {"path": "meteor"},
    "rouge",
    {"path": "sacrebleu"}
]

_HF_METRICS = "competition_math"


@pytest.fixture(scope="package")
def predictions():
    return ["There is a cat on the mat.", "Look! a wonderful day."]


@pytest.fixture(scope="package")
def references():
    return ["The cat is playing on the mat.", "Today is a wonderful day"]


@pytest.fixture
def predictions_sequence_classification():
    return [0, 2, 1, 0, 0, 1]


@pytest.fixture
def references_sequence_classification():
    return [0, 1, 2, 0, 1, 2]


@pytest.fixture
def multiple_predictions_sequence_classification():
    return [[0], [1, 2], [0], [1], [0], [1, 2]]


@pytest.fixture
def multiple_references_sequence_classification():
    return [[0, 2], [1, 0], [0, 1], [0], [0], [1, 2]]


@pytest.fixture(scope="function")
def inconsistent_predictions():
    return ["There is a cat on the mat."]


@pytest.fixture(scope="function")
def single_prediction_array():
    return [["the cat is on the mat"], ["Look! a wonderful day."]]


@pytest.fixture(scope="function")
def multiple_predictions_empty():
    return [
        [],
        ["Look! what a wonderful day, today.", "Today is a very wonderful day"],
    ]


@pytest.fixture(scope="function")
def multiple_references_empty():
    return [
        ["the cat is playing on the mat.", "The cat plays on the mat."],
        ["Today is a wonderful day", "The weather outside is wonderful."],
    ]


@pytest.fixture(scope="package")
def multiple_predictions():
    return [
        ["the cat is on the mat", "There is cat playing on mat"],
        ["Look! what a wonderful day, today.", "Today is a very wonderful day"],
    ]


@pytest.fixture(scope="package")
def multiple_references():
    return [
        ["the cat is playing on the mat.", "The cat plays on the mat."],
        ["Today is a wonderful day", "The weather outside is wonderful."],
    ]


@pytest.fixture(scope="module")
def nlgmetricverse_base():
    return NLGMetricverse(metrics=_TEST_METRICS)


@pytest.fixture(scope="function")
def nlgmetricverse_concurrent():
    return NLGMetricverse(metrics=_TEST_METRICS, run_concurrent=True)


@pytest.fixture(scope="function")
def nlgmetricverse_str():
    return NLGMetricverse(metrics=_STR_TEST_METRIC)


@pytest.fixture(scope="function")
def nlgmetricverse_list_str():
    return NLGMetricverse(metrics=_LIST_STR_TEST_METRICS)


@pytest.fixture(scope="function")
def nlgmetricverse_list_dict():
    return NLGMetricverse(metrics=_LIST_DICT_TEST_METRICS)


@pytest.fixture(scope="function")
def nlgmetricverse_list_mixed():
    return NLGMetricverse(metrics=_LIST_MIXED_TEST_METRICS)


@pytest.fixture(scope="function")
def nlgmetricverse_hf():
    return NLGMetricverse(metrics=_HF_METRICS)


def get_expected_output(prefix: Optional[str] = None):
    def json_load(path: str):
        with open(path, "r") as jf:
            content = json.load(jf)
        return content

    def wrapper(fn, *args, **kwargs):
        module_name = os.path.basename(inspect.getfile(fn)).replace(".py", "")
        path = os.path.join(EXPECTED_OUTPUTS, prefix, f"{module_name}.json")
        test_name = fn.__name__.replace("output_", "")
        fn.output = json_load(path)[test_name]
        return fn

    if prefix is None:
        prefix = ""
    return wrapper
