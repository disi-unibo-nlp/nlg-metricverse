import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tqdm import tqdm

from nlgmetricverse.utils.correlation import *


def metrics_correlation(
        predictions,
        references,
        metrics,
        correlation_measures=[CorrelationMeasures.Pearson]
):
    """
    Calculates the correlation between different metrics, with the possibility of choosing
    between several correlation techniques.

    :param predictions: List of predictions
    :param references: List of references
    :param metrics: List of metrics
    :param correlation_measures: The correlation technique to apply
    """
    if len(metrics) < 2:
        raise ValueError("'metric' length must be at least 2")
    if not isinstance(predictions, list) and not isinstance(references, list):
        raise Exception("predictions and references must be of type list")
    matrix_res = np.zeros((len(metrics), len(metrics)))

    scores = calculate_scores(predictions, references, metrics, human_flag=False)

    results = []
    pvalues = []
    pvalue_results = np.zeros((len(metrics), len(metrics)))
    """
    Creates a matrix to store correlation coefficients and p-values, with a for 
    loop iterates through different correlation measures and through pairs of metrics.
    Computes then correlation and store in the matrix, creates a mask to visualize the 
    lower triangle of the correlation matrix and a heatmap to visualize the correlation matrix.
    Lastly, appends the correlation matrix and the p-value matrix to the results and pvalues lists.
    """
    for correlation_measure in tqdm(correlation_measures, desc="Calculating correlation"):
        for i, metricA in enumerate(metrics):
            for j, metricB in enumerate(metrics):
                matrix_res[i][j], pvalue_results[i][j] = compute_correlation(scores[metricA], scores[metricB], correlation_measure)
        mask = np.triu(np.ones_like(matrix_res, dtype=bool))
        sns.heatmap(np.tril(matrix_res, -1), xticklabels=metrics, yticklabels=metrics, annot=True, mask=mask,
                           cmap="Blues")

        results.append(np.tril(matrix_res, -1))
        pvalues.append(np.tril(pvalue_results, -1))

        plt.title("Metric scores correlation")
        plt.tight_layout()
    return results, pvalues
