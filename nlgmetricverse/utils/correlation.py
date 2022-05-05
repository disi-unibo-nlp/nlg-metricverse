from nlgmetricverse import load_metric, Nlgmetricverse


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
