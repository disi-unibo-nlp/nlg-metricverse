import evaluate
import numpy as np
import pytest

from tests.nlgmetricverse.conftest import get_expected_output
from tests.utils import assert_almost_equal_dict


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_concurrent():
    return output_evaluate_concurrent.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate():
    return output_evaluate.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_str_input():
    return output_evaluate_str_input.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_list_str_input():
    return output_evaluate_list_str_input.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_list_dict_input():
    return output_evaluate_list_dict_input.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_list_mixed_input():
    return output_evaluate_list_mixed_input.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_corpus():
    return output_evaluate_corpus.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_multiple_predictions_empty():
    return output_evaluate_multiple_predictions_empty.output


@pytest.fixture
@get_expected_output(prefix=None)
def output_evaluate_multiple_predictions():
    return output_evaluate_multiple_predictions.output


def test_evaluate_concurrent(predictions, references, nlgmetricverse_concurrent, output_evaluate_concurrent):
    scores = nlgmetricverse_concurrent(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_evaluate_concurrent)


def test_evaluate_no_input(predictions, references, nlgmetricverse_base):
    with pytest.raises(TypeError):
        nlgmetricverse_base(predictions=predictions)
        nlgmetricverse_base(references=references)
        nlgmetricverse_base()


def test_evaluate_inconsistent_input(inconsistent_predictions, references, nlgmetricverse_base):
    # Different length
    with pytest.raises(ValueError):
        nlgmetricverse_base(predictions=inconsistent_predictions, references=references)


def test_evaluate(predictions, references, nlgmetricverse_base, output_evaluate):
    scores = nlgmetricverse_base(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_evaluate)


def test_evaluate_str_input(predictions, references, nlgmetricverse_str, output_evaluate_str_input):
    scores = nlgmetricverse_str(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_evaluate_str_input)


def test_evaluate_list_str_input(predictions, references, nlgmetricverse_list_str, output_evaluate_list_str_input):
    scores = nlgmetricverse_list_str(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_evaluate_list_str_input)


def test_evaluate_list_dict_input(predictions, references, nlgmetricverse_list_dict, output_evaluate_list_dict_input):
    scores = nlgmetricverse_list_dict(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_evaluate_list_dict_input)


def test_evaluate_list_mixed_input(predictions, references, nlgmetricverse_list_mixed, output_evaluate_list_mixed_input):
    scores = nlgmetricverse_list_mixed(predictions=predictions, references=references)
    assert_almost_equal_dict(actual=scores, desired=output_evaluate_list_mixed_input)


def test_evaluate_corpus(single_prediction_array, multiple_references, nlgmetricverse_base, output_evaluate_corpus):
    scores = nlgmetricverse_base(predictions=single_prediction_array, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_evaluate_corpus)


def test_evaluate_multiple_predictions_empty(
    multiple_predictions_empty, multiple_references_empty, nlgmetricverse_base, output_evaluate_multiple_predictions_empty
):
    scores = nlgmetricverse_base(predictions=multiple_predictions_empty, references=multiple_references_empty)
    assert_almost_equal_dict(actual=scores, desired=output_evaluate_multiple_predictions_empty)


def test_evaluate_multiple_predictions(
    multiple_predictions, multiple_references, nlgmetricverse_base, output_evaluate_multiple_predictions
):
    scores = nlgmetricverse_base(predictions=multiple_predictions, references=multiple_references)
    assert_almost_equal_dict(actual=scores, desired=output_evaluate_multiple_predictions)


def test_reduce_fn(predictions, references, nlgmetricverse_base):
    _reduce_fn = np.mean
    _non_reduce_fn = np.exp
    scores = nlgmetricverse_base(predictions=predictions, references=references, reduce_fn=_reduce_fn)

    assert all([scores[metric.resulting_name] is not None for metric in nlgmetricverse_base.metrics])

    with pytest.raises(ValueError):
        nlgmetricverse_base(predictions=predictions, references=references, reduce_fn=_non_reduce_fn)


def test_load_metric():
    from nlgmetricverse import load_metric
    from nlgmetricverse.metrics._core import Metric as NlgmetricverseMetric

    assert isinstance(load_metric("chrf"), NlgmetricverseMetric)
    assert isinstance(load_metric("squad_v2"), evaluate.Metric)

    with pytest.raises(FileNotFoundError):
        load_metric("abcdefgh")
