import random

import matplotlib.pyplot as plt
import numpy as np

from nlgmetricverse.utils.correlation import *
from nlgmetricverse.utils.benchmarks.get_wmt17_sys_results import *
from nlgmetricverse import NLGMetricverse, load_metric


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
            single_metric_scores = []
            res = []
            single_metric_scorer = NLGMetricverse(metrics=load_metric(metric))
            for i, pred in enumerate(predictions):
                single_metric_score = single_metric_scorer(predictions=[pred], references=[references[i]])
                single_metric_scores.append(single_metric_score)
                for single_score in single_metric_score:
                    if isinstance(single_metric_score[single_score], dict):
                        if metric == "rouge":
                            mean = single_metric_score[single_score]["rouge1"] + \
                                   single_metric_score[single_score]["rouge2"] + single_metric_score[single_score][
                                       "rougeL"]
                            mean = mean / 3
                            res.append(mean)
                        else:
                            res.append(single_metric_score[single_score]["score"])
            scores[metric] = map_score_with_metric_bounds(metric, res)

        results = []
        for metric in metrics:
            metric_scores = []
            for correlation_measure in correlation_measures:
                statistic, pvalue = compute_correlation(scores[metric], human_scores, correlation_measure)
                statistic = map_range(statistic, -1, 1, 0, 1)
                metric_scores.append(statistic)
            results.append(np.mean(metric_scores))

        bar_list = plt.bar(metrics, results)

        for bar in bar_list:
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            bar.set_color(color)

        plt.xticks(np.arange(len(metrics)), metrics)
        plt.xlabel("Metrics")
        plt.ylabel("Scores")
        plt.title("Metric-human correlation")
        plt.legend()
        # plt.show()'''
        return results
