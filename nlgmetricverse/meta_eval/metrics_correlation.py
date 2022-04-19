import os

import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

import nlgmetricverse
from nlgmetricverse import data_loader


def pearson_and_spearman(
        metrics=None,
        method="read_lines",
        predictions=os.getcwd() + "/correlation/predictions",
        references=os.getcwd() + "/correlation/references"
):
    if metrics is None:
        metrics = [
            "bertscore",
            "bleu",
            "chrf",
            "meteor",
            "rouge",
            "sacrebleu",
            "ter",
            "wer"
        ]
    if len(metrics) < 2:
        raise ValueError("'metric' length must be at least 2")
    matrixResP = np.zeros((len(metrics), len(metrics)))
    matrixResS = np.zeros((len(metrics), len(metrics)))
    scores = {}
    for metric in metrics:
        score = scores_single_metric(metric=metric, predictions=predictions, references=references, method=method)
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
            matrixResP[i][j] = pearsonr(scores[metricA], scores[metricB])[0]
            matrixResS[i][j] = spearmanr(scores[metricA], scores[metricB])[0]
    matrix_to_plot(matrixResP, metrics)
    matrix_to_plot(matrixResS, metrics)

    return {
        "pearson": np.tril(matrixResP, -1),
        "spearman": np.tril(matrixResS, -1)
    }


def scores_single_metric(metric, predictions, references, method):
    scorer = nlgmetricverse.load_metric(metric)
    dl = data_loader.DataLoader(predictions=predictions, references=references, method=method)
    preds = dl.get_predictions()
    refs = dl.get_references()
    scores = []
    res = []
    for i, pred in enumerate(preds):
        if metric == "bleu":
            score = scorer.compute(predictions=[pred], references=[refs[i]], max_order=1)
        else:
            score = scorer.compute(predictions=[pred], references=[refs[i]])
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
