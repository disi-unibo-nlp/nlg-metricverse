import matplotlib.pyplot as plt

from nlgmetricverse import data_loader, DataLoaderStrategies
from nlgmetricverse.utils.correlation import *


def times_correlation(
        predictions,
        references,
        metrics,
        strategy=DataLoaderStrategies.ReadLines
):
    if not isinstance(predictions, list) and not isinstance(references, list):
        raise Exception("predictions and references must be of type list")
    times = {}
    for metric in metrics:
        checked_metric = check_metric(metric)
        scorer = NLGMetricverse(metrics=checked_metric)
        results = scorer(predictions=predictions, references=references)
        times[metric] = results["total_time_elapsed"]

    plt.bar(times.keys(), times.values(), color='b')
    plt.title("Time correlation between metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Time [sec]")
    plt.show()

    return times
