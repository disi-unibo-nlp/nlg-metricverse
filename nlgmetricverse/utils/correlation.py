from nlgmetricverse import load_metric


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
