from collections import Counter
from nlgmetricverse import data_loader


def repetitiveness(predictions, references, method):
    """
    Count how many words are repeated in predictions.

    :param predictions: Predictions.
    :param references: References.
    :param method: Method to analyse inputs. Can be "no_new_line" or "read_lines".
    :return: (float) repetitiveness.
    """
    dl = data_loader.DataLoader(predictions=predictions, references=references, method=method)
    res_predictions = dl.get_predictions()
    if isinstance(res_predictions[0], list):
        inputList = []
        for prediction in res_predictions:
            inputList += prediction
        result = compute_repetitiveness(inputList)
    else:
        result = compute_repetitiveness(res_predictions)
    return result


def compute_repetitiveness(predictions):
    counter = 0
    for candidate in predictions:
        monograms = candidate.split(" ")
        n_words = len(monograms)
        m_counted = Counter(monograms)
        for ngram in m_counted.values():
            if ngram > 1:
                counter = counter + 1  # if a word  that repeats itself is found
        counter = counter + n_words
    return counter / len(predictions)
