import os

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import seaborn as sns
import matplotlib.pyplot as plt

from nlgmetricverse import data_loader, Nlgmetricverse, load_metric


def pearson_and_spearman(
        metrics=None,
        method="read_lines",
        technique="pearson",
        predictions=os.getcwd() + "/correlation/predictions",
        references=os.getcwd() + "/correlation/references"
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
    matrix_to_plot(matrixRes, metrics)

    return np.tril(matrixRes, -1)


def scores_single_metric(metric, predictions, references):
    scores = []
    res = []
    METRIC = check_metric(metric)
    scorer = Nlgmetricverse(metrics=METRIC)
    for i, pred in enumerate(predictions):
        score = scorer(predictions=[pred], references=[references[i]])
        scores.append(score)
        for single_score in score:
            if isinstance(score[single_score], dict):
                if metric == "rouge":
                    res.append(score[single_score]["rouge1"])
                else:
                    res.append(score[single_score]["score"])
    return res


def map_range(value, left_min, left_max, right_min, right_max):
    leftSpan = left_max - left_min
    rightSpan = right_max - right_min
    valueScaled = float(value - left_min) / float(leftSpan)
    return right_min + (valueScaled * rightSpan)


def matrix_to_plot(matrix, metrics):
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(np.tril(matrix, -1), xticklabels=metrics, yticklabels=metrics, annot=True, mask=mask, cmap="Blues")
    plt.show()


def check_metric(metric):
    if metric == "bleu":
        METRIC = [load_metric("bleu", resulting_name="bleu1", compute_kwargs={"max_order": 1})]
    else:
        METRIC = [load_metric(metric)]
    return METRIC
