import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau

from nlgmetricverse import data_loader
from nlgmetricverse.utils.correlation import *


def metrics_correlation(
        predictions,
        references,
        metrics=None,
        method="read_lines",
        technique="pearson"
):
    if metrics is None:
        metrics = [
            "bleu",
            "chrf",
            "meteor",
            "rouge",
            "sacrebleu",
            "ter"
        ]
    if len(metrics) < 2:
        raise ValueError("'metric' length must be at least 2")
    if not isinstance(predictions, list) and not isinstance(references, list):
        dl = data_loader.DataLoader(predictions=predictions, references=references, method=method)
        predictions = dl.get_predictions()
        references = dl.get_references()
    matrixRes = np.zeros((len(metrics), len(metrics)))
    scores = {}
    for metric in metrics:
        score = scores_single_metric(metric=metric, predictions=predictions, references=references)
        if metric == "bartscore":
            mapped_score = map_range(score, float('-inf'), 0, 0, 1)
        elif metric == "nist":
            mapped_score = map_range(score, 0, 10, 0, 1)
        elif metric == "perplexity":
            mapped_score = map_range(score, float('inf'), 1, 0, 1)
        else:
            mapped_score = score
        scores[metric] = mapped_score
    for i, metricA in enumerate(metrics):
        for j, metricB in enumerate(metrics):
            if technique == "pearson":
                matrixRes[i][j] = pearsonr(scores[metricA], scores[metricB])[0]
            elif technique == "spearman":
                matrixRes[i][j] = spearmanr(scores[metricA], scores[metricB])[0]
            elif technique == "kendalltau":
                matrixRes[i][j] = kendalltau(scores[metricA], scores[metricB])[0]
    mask = np.triu(np.ones_like(matrixRes, dtype=bool))
    sns.heatmap(np.tril(matrixRes, -1), xticklabels=metrics, yticklabels=metrics, annot=True, mask=mask, cmap="Blues")
    plt.title("Metric scores correlation")
    plt.show()

    return np.tril(matrixRes, -1)
