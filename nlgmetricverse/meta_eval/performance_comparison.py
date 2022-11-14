import matplotlib.pyplot as plt

from nlgmetricverse import NLGMetricverse, load_metric
from nlgmetricverse.utils.correlation import *


def times_correlation(
        predictions,
        references,
        metrics
):
    """
    Compare the computational time of metrics.

    :param predictions: List of predictions
    :param references: List of references
    :param metrics: List of metrics
    """
    if not isinstance(predictions, list) and not isinstance(references, list):
        raise Exception("predictions and references must be of type list")
    times = {}
    for metric in metrics:
        scorer = NLGMetricverse(metrics=load_metric(metric))
        results = scorer(predictions=predictions, references=references)
        times[metric] = results["total_time_elapsed"]

    plt.bar(times.keys(), times.values(), color='b')
    plt.title("Time correlation between metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Time [sec]")
    # plt.show()

    return times
