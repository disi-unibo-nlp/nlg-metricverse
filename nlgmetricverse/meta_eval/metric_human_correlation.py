import random

import matplotlib.pyplot as plt

from nlgmetricverse.utils.correlation import *
from nlgmetricverse.utils.benchmarks.get_wmt17_sys_results import *


def metric_human_correlation(
        predictions,
        references,
        metrics,
        human_scores=Benchmarks.WMT17,
        correlation_measures=None,
):
    """
    Calculates the correlation between human scores and those of the metrics, with the possibility of choosing
    between several correlation techniques.

    :param predictions: List of predictions
    :param references: List of references
    :param metrics: List of metrics
    :param human_scores: Can be a list of personal scores or can choose a public benchmark, WMT17 by default
    :param correlation_measures: The correlation technique to apply
    """
    if correlation_measures is None:
        correlation_measures = [CorrelationMeasures.Pearson]
    if not isinstance(human_scores, list):
        if human_scores == Benchmarks.WMT17:
            # Generate ad DB with human scores from WMT
            # print("Downloading wmt17 data, first time might take a bit...")
            wmt17_download_data()
            # print("Download completed")
            # print("Evaluating...")
            data = get_wmt17_sys_results()
            # print("Evaluation completed, you can see the results in the 'wmt17' folder")
            return data
    else:
        # Using personal scores
        scores = {}
        for metric in metrics:
            if not isinstance(predictions, list) and not isinstance(references, list):
                raise Exception("predictions and references must be of type list")
            score = scores_single_metric(metric=metric, predictions=predictions, references=references)
            mapped_score = []
            for i, single_score in enumerate(score):
                if metric == "bartscore":
                    mapped_score.append(map_range(single_score, -7, 0, 0, 1))
                elif metric == "nist":
                    mapped_score.append(map_range(single_score, 0, 10, 0, 1))
                elif metric == "perplexity":
                    mapped_score.append(map_range(single_score, float('inf'), 1, 0, 1))
                else:
                    mapped_score.append(single_score)
            scores[metric] = mapped_score
        results = []
        for correlation_measure in correlation_measures:
            correlation_results = {}
            for i, metric in enumerate(metrics):
                correlation_results[metric] = compute_correlation(scores[metric], human_scores, correlation_measure)
                correlation_results[metric] = map_range(correlation_results[metric], -1, 1, 0, 1)
            bar_list = plt.bar(list(correlation_results.keys()), correlation_results.values(), label=correlation_measure)
            for bar in bar_list:
                r = random.random()
                b = random.random()
                g = random.random()
                color = (r, g, b)
                bar.set_color(color)
            results.append(correlation_results)
            plt.xticks(np.arange(len(correlation_measures)), correlation_measures)
            plt.xlabel("Correlation measure")
            plt.ylabel("Scores")
            plt.title("Metric-human correlation")
            plt.legend()
            plt.show()
        return results
