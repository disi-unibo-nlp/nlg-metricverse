import os
from scipy.stats import pearsonr, spearmanr

from nlgmetricverse import Nlgmetricverse, data_loader


def pearson_and_spearman(
        metrics=None,
        method="read_lines",
        predictions=os.getcwd() + "../tests/predictions",
        references=os.getcwd() + "../tests/references"
):
    if metrics is None:
        metrics = [
            "bleu",
            "meteor"
        ]
    res1 = scores_single_metric(metric="bertscore", predictions=predictions, references=references, method=method)
    res2 = scores_single_metric(metric="meteor", predictions=predictions, references=references, method=method)
    pearson_corr = pearsonr(res1, res2)[0]
    spearman_corr = spearmanr(res1, res2)[0]
    print("res1: ", res1)
    print("res2: ", res2)
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr
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
                res.append(score[single_score]["score"])
    return res
