from nlgmetricverse import data_loader


def abstractness(predictions, references, method, n=1):
    """
    Count abstractness of the text.

    :param predictions: Predictions.
    :param references: References.
    :param method: Method to analyse inputs. Can be "no_new_line" or "read_lines".
    :param n: Length of monograms, 1 by default.
    :return: (float) abstractness.
    """
    dl = data_loader.DataLoader(predictions=predictions, references=references, method=method)
    res_predictions = dl.get_predictions()
    res_references = dl.get_references()

    if isinstance(res_predictions[0], list):
        predList = []
        refList = []
        for pred in res_predictions:
            predList += pred
        for ref in res_references:
            refList += ref
        result = compute_abstractness(refList, predList, n)
    else:
        result = compute_abstractness(res_references, res_predictions, n)
    return result


def compute_abstractness(res_references, res_predictions, n):
    total_match = 0
    n_words = 0
    for reference, candidate in zip(res_references, res_predictions):
        match = 0
        monograms = candidate.split(" ")
        n_words = n_words + len(monograms)  # count all words in test set
        if n > len(monograms):
            return "Not possible to create " + str(n) + "-grams, too many few words"
        for w2 in ngrams(monograms, n):
            substr = " ".join(w2)
            if substr not in reference:
                match = match + 1
        # n_words=n_words+1 #counter for total n-gram number
        total_match = total_match + match
    return total_match / n_words


def ngrams(tokens, n):
    """
    Provides an iterable object of n-gram.

    :param tokens: Monograms.
    :param n: Length of the iterable object.
    :return: Iterable object of n-gram.
    """
    ngram = []
    for token in tokens:
        if len(ngram) < n:
            ngram.append(token)
        else:
            yield ngram
            ngram.pop(0)
            ngram.append(token)
    if len(ngram) == n:
        yield ngram

