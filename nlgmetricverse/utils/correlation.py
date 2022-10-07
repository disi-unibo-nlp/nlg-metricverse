from enum import Enum
from nlgmetricverse import load_metric, NLGMetricverse


def map_range(value, left_min, left_max, right_min, right_max):
    leftSpan = left_max - left_min
    rightSpan = right_max - right_min
    valueScaled = float(value - left_min) / float(leftSpan)
    return right_min + (valueScaled * rightSpan)


def check_metric(metric):
    if metric == "bleu":
        METRIC = [load_metric("bleu", resulting_name="bleu1", compute_kwargs={"max_order": 1})]
    else:
        METRIC = [load_metric(metric)]
    return METRIC


def scores_single_metric(metric, predictions, references):
    scores = []
    res = []
    checked_metric = check_metric(metric)
    scorer = NLGMetricverse(metrics=checked_metric)
    for i, pred in enumerate(predictions):
        score = scorer(predictions=[pred], references=[references[i]])
        scores.append(score)
        for single_score in score:
            if isinstance(score[single_score], dict):
                if metric == "rouge":
                    mean = score[single_score]["rouge1"] + score[single_score]["rouge2"] + score[single_score]["rougeL"]
                    mean = mean / 3
                    res.append(mean)
                else:
                    res.append(score[single_score]["score"])
    return res


class CorrelationMeasures(Enum):
    Pearson = 1
    Spearman = 2
    KendallTau = 3


class Benchmarks(Enum):
    WMT17 = 1
