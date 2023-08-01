# import pytest

# from nlgmetricverse import NLGMetricverse
# from nlgmetricverse.metrics import AutoMetric
# from tests.nlgmetricverse.conftest import get_expected_output
# from tests.utils import assert_almost_equal_dict


# @pytest.fixture(scope="module")
# def nlgmetricverse_comet():
#     metric = AutoMetric.load(
#         "comet",
#         config_name="wmt21-cometinho-da",
#         compute_kwargs={"gpus": 0, "num_workers": 0, "progress_bar": False, "batch_size": 2},
#     )
#     return NLGMetricverse(metrics=metric)


# @pytest.fixture(scope="module")
# def comet_sources():
#     return ["Die Katze spielt auf der Matte.", "Heute ist ein wunderbarer Tag."]


# @pytest.fixture
# @get_expected_output(prefix="metrics")
# def output_basic():
#     return output_basic.output


# @pytest.fixture
# @get_expected_output(prefix="metrics")
# def output_multiple_ref():
#     return output_multiple_ref.output


# @pytest.fixture
# @get_expected_output(prefix="metrics")
# def output_multiple_pred_multiple_ref():
#     return output_multiple_pred_multiple_ref.output


# def test_basic(comet_sources, predictions, references, nlgmetricverse_comet, output_basic):
#     scores = nlgmetricverse_comet(sources=comet_sources, predictions=predictions, references=references)
#     assert_almost_equal_dict(actual=scores, desired=output_basic)


# def test_multiple_ref(comet_sources, predictions, multiple_references, nlgmetricverse_comet, output_multiple_ref):
#     scores = nlgmetricverse_comet(sources=comet_sources, predictions=predictions, references=multiple_references)
#     assert_almost_equal_dict(actual=scores, desired=output_multiple_ref)


# def test_multiple_pred_multiple_ref(
#     comet_sources, multiple_predictions, multiple_references, nlgmetricverse_comet, output_multiple_pred_multiple_ref
# ):
#     scores = nlgmetricverse_comet(sources=comet_sources, predictions=multiple_predictions, references=multiple_references)
#     assert_almost_equal_dict(actual=scores, desired=output_multiple_pred_multiple_ref)
