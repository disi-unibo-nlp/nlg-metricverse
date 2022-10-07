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
