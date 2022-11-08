import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nlgmetricverse.utils.correlation import *


def metrics_correlation(
        predictions,
        references,
        metrics,
        correlation_measures=None
):
    """
    Calculates the correlation between different metrics, with the possibility of choosing
    between several correlation techniques.

    :param predictions: List of predictions
    :param references: List of references
    :param metrics: List of metrics
    :param correlation_measures: The correlation technique to apply
    """
    if correlation_measures is None:
        correlation_measures = [CorrelationMeasures.Pearson]
    if len(metrics) < 2:
        raise ValueError("'metric' length must be at least 2")
    if not isinstance(predictions, list) and not isinstance(references, list):
        raise Exception("predictions and references must be of type list")
    matrix_res = np.zeros((len(metrics), len(metrics)))
    scores = {}
    for metric in metrics:
        single_metric_scores = []
        res = []
        single_metric_scorer = NLGMetricverse(metrics=metric)
        for i, pred in enumerate(predictions):
            single_metric_score = single_metric_scorer(predictions=[pred], references=[references[i]])
            single_metric_scores.append(single_metric_score)
            for single_score in single_metric_score:
                if isinstance(single_metric_score[single_score], dict):
                    if metric == "rouge":
                        mean = single_metric_score[single_score]["rouge1"] + single_metric_score[single_score][
                            "rouge2"] + single_metric_score[single_score][
                                   "rougeL"]
                        mean = mean / 3
                        res.append(mean)
                    else:
                        res.append(single_metric_score[single_score]["score"])
        scores[metric] = map_score_with_metric_bounds(metric, res)

    results = []
    # p_values = []
    for correlation_measure in correlation_measures:
        for i, metricA in enumerate(metrics):
            for j, metricB in enumerate(metrics):
                matrix_res[i][j] = compute_correlation(scores[metricA], scores[metricB], correlation_measure)
        mask = np.triu(np.ones_like(matrix_res, dtype=bool))
        sns.heatmap(np.tril(matrix_res, -1), xticklabels=metrics, yticklabels=metrics, annot=True, mask=mask,
                           cmap="Blues")

        results.append(np.tril(matrix_res, -1))
        # p_values.append()
        plt.title("Metric scores correlation")
        plt.tight_layout()
        plt.show()

    return results
