import pytest

from nlgmetricverse import NLGMetricverse
from nlgmetricverse.metrics import AutoMetric
from tests.nlgmetricverse.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict


@pytest.fixture(scope="module")
def nlgmetricverse_cer():
    metric = AutoMetric.load("cer")
    return NLGMetricverse(metrics=metric)


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic():
    return output_basic.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_basic_concat():
    return output_basic_concat.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_ref():
    return output_multiple_ref.output


@pytest.fixture
@get_expected_output(prefix="metrics")
def output_multiple_pred_multiple_ref():
    return output_multiple_pred_multiple_ref.output


def test_basic(predictions, references, nlgmetricverse_cer, output_basic):
    scores = nlgmetricverse_cer(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_basic)


def test_basic_concat(predictions, references, nlgmetricverse_cer, output_basic_concat):
    scores = nlgmetricverse_cer(predictions=predictions, references=references, concatenate_texts=True)
    assert_almost_equal_dict(actual=scores, desired=output_basic_concat)


def test_multiple_ref(predictions, multiple_references, nlgmetricverse_cer, output_multiple_ref):
    scores = nlgmetricverse_cer(predictions=predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_ref)


def test_multiple_pred_multiple_ref(
    multiple_predictions, multiple_references, nlgmetricverse_cer, output_multiple_pred_multiple_ref
):
    scores = nlgmetricverse_cer(predictions=multiple_predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref)