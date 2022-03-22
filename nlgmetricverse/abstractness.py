from nlgmetricverse import data_loader


def abstractness(predictions, references, method, n=1):
    dl = data_loader.DataLoader(predictions=predictions, references=references, method=method)
    res_predictions = dl.get_predictions()
    res_references = dl.get_references()

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


def ngrams(tokens, n):  # provides an iterable object of n-gram
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
