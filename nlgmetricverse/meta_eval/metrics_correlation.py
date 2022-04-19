import os
from scipy.stats import pearsonr, spearmanr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from nlgmetricverse import Nlgmetricverse, data_loader


def pearson_and_spearman(
        metrics=None,
        method="read_lines",
        predictions=os.getcwd() + "/correlation/predictions",
        references=os.getcwd() + "/correlation/references"
):
    if metrics is None:
        metrics = [
            "bertscore",
            "meteor",
            "rouge",
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
        metrics.remove(metricA)
        for j, metricB in enumerate(metrics):
            resA = scores[metricA]
            resB = scores[metricB]
            matrixResP[i][j] = pearsonr(resA, resB)[0]
            matrixResS[i][j] = spearmanr(resA, resB)[0]
    print("matrixResP: ", matrixResP)
    print("matrixResS: ", matrixResS)

    sns.heatmap(matrixResP, annot=True)
    plt.show()
    sns.heatmap(matrixResS, annot=True)
    plt.show()


    return {
        "pearson": matrixResP,
        "spearman": matrixResS
    }


def scores_single_metric(metric, predictions, references, method):
    scorer = Nlgmetricverse(metrics=[metric])
    dl = data_loader.DataLoader(predictions=predictions, references=references, method=method)
    preds = dl.get_predictions()
    refs = dl.get_references()
    scores = []
    res = []
    for i, pred in enumerate(preds):
        score = scorer(predictions=pred, references=refs[i], method=method)
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
