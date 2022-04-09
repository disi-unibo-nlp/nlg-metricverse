import os
from scipy.stats import pearsonr, spearmanr

from nlgmetricverse import Nlgmetricverse


def pearson_and_spearman(
        values,
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
    scorer = Nlgmetricverse(metrics=metrics)
    scores = scorer(predictions=predictions, references=references)
    res = []
    for score in scores:
        if isinstance(scores[score], dict):
            res.append(scores[score]["score"])
    print(res)
    pearson_corr = pearsonr(values, res)[0]
    spearman_corr = spearmanr(values, res)[0]
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr
    }
