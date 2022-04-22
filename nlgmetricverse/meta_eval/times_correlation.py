import matplotlib.pyplot as plt

from nlgmetricverse import data_loader, Nlgmetricverse
from nlgmetricverse.utils.correlation import *


def times_correlation(
        predictions,
        references,
        metrics=None,
        method="read_lines"
):
    if metrics is None:
        metrics = [
            "bartscore",
            "bleu",
            "chrf",
            "meteor",
            "rouge",
            "sacrebleu",
            "ter",
            "wer"
        ]
    if not isinstance(predictions, list) and not isinstance(references, list):
        dl = data_loader.DataLoader(predictions=predictions, references=references, method=method)
        predictions = dl.get_predictions()
        references = dl.get_references()
    times = {}
    for metric in metrics:
        METRIC = check_metric(metric)
        scorer = Nlgmetricverse(metrics=METRIC)
        results = scorer(predictions=predictions, references=references)
        times[metric] = results["total_time_elapsed"]

    plt.bar(times.keys(), times.values(), color='b')
    plt.title("Time correlation between metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Time [sec]")
    plt.show()

    return times
