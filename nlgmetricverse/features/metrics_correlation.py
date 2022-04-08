import os
from scipy.stats import pearsonr, spearmanr

from nlgmetricverse import Nlgmetricverse
from nlgmetricverse import data_loader


def pearson_and_spearman(
        values,
        method="read_lines",
        predictions=os.getcwd() + "../tests/predictions",
        references=os.getcwd() + "../tests/references"
):
    metrics = [
        "bleu",
        "meteor"
    ]
    dl = data_loader.DataLoader(predictions=predictions, references=references, method=method)
    preds = dl.get_predictions()
    refs = dl.get_references()
    for metric in metrics:
        score = scores_single_metric(preds, refs, metric)
    pearson_corr = pearsonr(score, score)[0]
    spearman_corr = spearmanr(score, score)[0]
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr
    }


def scores_single_metric(predictions, references, metric):
    scores = []
    for i, pred in enumerate(predictions):
        scorer = Nlgmetricverse(metrics=[metric])
        score = scorer(predictions=pred, references=references[i])
        if metric in score:
            scores.append(score[metric]["score"])
        print(score)
    return scores
