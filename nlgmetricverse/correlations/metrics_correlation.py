import os
from scipy.stats import pearsonr, spearmanr
import numpy as np

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
            "rouge",
            "meteor"
        ]
    matrixResP = np.zeros((len(metrics), len(metrics)))
    matrixResS = np.zeros((len(metrics), len(metrics)))
    for i, metricA in enumerate(metrics):
        metrics.remove(metricA)
        for j, metricB in enumerate(metrics):
            res1 = scores_single_metric(metric=metricA, predictions=predictions, references=references, method=method)
            res2 = scores_single_metric(metric=metricB, predictions=predictions, references=references, method=method)
            matrixResP[i][j] = pearsonr(res1, res2)[0]
            matrixResS[i][j] = spearmanr(res1, res2)[0]
    print("matrixResP: ", matrixResP)
    print("matrixResS: ", matrixResS)
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
