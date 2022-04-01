import numpy as np

from nlgmetricverse import data_loader
import syllables


def readability(predictions, references, method):
    """
    Calc readability of the text.

    :param predictions: Predictions.
    :param references: References.
    :param method: Method to analyse inputs. Can be "no_new_line" or "read_lines".
    :return: (float) readability.
    """
    dl = data_loader.DataLoader(predictions=predictions, references=references, method=method)
    res_predictions = dl.get_predictions()

    if isinstance(res_predictions[0], list):
        res = []
        for prediction in res_predictions:
            score = compute_readability(prediction)
            res.append(score)
        result = np.mean(res)
    else:
        result = compute_readability(res_predictions)
    return result


def compute_readability(predictions):
    total_words = 0
    total_sentences = 0
    total_syllables = 0
    result = 0
    print(predictions)

    total_sentences = len(predictions)
    for sentence in predictions:
        total_words += len(sentence.split())
        total_syllables += syllables.estimate(sentence)
    print(total_words)
    print(total_sentences)
    print(total_syllables)
    result = 206.835 - 1.015 * (total_words / total_sentences) - 86.6 * (total_syllables / total_words)
    return result
