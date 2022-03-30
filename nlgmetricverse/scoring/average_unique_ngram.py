from nltk import ngrams

from nlgmetricverse import data_loader


def average_unique_ngram(predictions, references, method, n=1):
    """
    Count average unique n-grams' ratio.

    :param predictions: Predictions.
    :param references: References.
    :param method: Method to analyse inputs. Can be "no_new_line" or "read_lines".
    :param n: Length of grams, 1 by default.
    :return: (float) abstractness.
    """
    dl = data_loader.DataLoader(predictions=predictions, references=references, method=method)
    res_predictions = dl.get_predictions()

    if isinstance(res_predictions[0], list):
        inputList = []
        for prediction in res_predictions:
            inputList += prediction
        result = compute_average_unique_ngram(inputList, n)
    else:
        result = compute_average_unique_ngram(res_predictions, n)
    return result


def compute_average_unique_ngram(predictions, n):
    n_grams_count = 0
    unique_n_grams_count = 0

    for candidate in predictions:
        n_grams = list(ngrams(candidate.split(), n))
        for _ in n_grams:
            n_grams_count += 1
        unique_n_grams = remove_duplicates(n_grams)
        unique_n_grams_count += len(unique_n_grams)
    return unique_n_grams_count / n_grams_count


def remove_duplicates(lst):
    return [t for t in (set(tuple(i) for i in lst))]
