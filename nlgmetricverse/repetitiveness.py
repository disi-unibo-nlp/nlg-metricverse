from collections import Counter
from nlgmetricverse import data_loader


def repetitiveness(predictions, references, method):
    """
    Count how many words are repeated in predictions.

    :param predictions: Predictions.
    :param references: References.
    :param method: Necessary for data_loader but doesn't affect the count.
    :return: (int) count.
    """
    dl = data_loader.DataLoader(predictions=predictions, references=references, method=method)
    res_predictions = dl.get_predictions()
    counter = 0
    for candidate in res_predictions:
        monograms = candidate.split(" ")
        n_words = len(monograms)
        m_counted = Counter(monograms)
        for ngram in m_counted.values():
            if ngram > 1:
                counter = counter + 1  # if a word  that repeats itself is found
        counter = counter + n_words
    return counter / len(res_predictions)
