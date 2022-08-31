import pytest

from nlgmetricverse import NLGMetricverse
from nlgmetricverse.metrics import AutoMetric
from tests.nlgmetricverse.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict


@pytest.fixture(scope="module")
def nlgmetricverse_ter():
    metric = AutoMetric.load("ter", compute_kwargs={"normalized": True})
    return NLGMetricverse(metrics=metric)


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic():
    return output_basic.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_ref():
    return output_multiple_ref.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_pred_multiple_ref():
    return output_multiple_pred_multiple_ref.output


def test_basic(predictions, references, nlgmetricverse_ter, output_basic):
    scores = nlgmetricverse_ter(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_basic)


def test_multiple_ref(predictions, multiple_references, nlgmetricverse_ter, output_multiple_ref):
    scores = nlgmetricverse_ter(predictions=predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref)


def test_multiple_pred_multiple_ref(
    multiple_predictions, multiple_references, nlgmetricverse_ter, output_multiple_pred_multiple_ref
):
    scores = nlgmetricverse_ter(predictions=multiple_predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref)
