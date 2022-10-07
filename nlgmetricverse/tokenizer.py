from typing import List, Tuple

from nlgmetricverse.collator import Collator
from nlgmetricverse.utils.string import normalize_text


class DefaultTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return normalize_text(text).split()


class TokenizerWrapper:
    """
    Wraps the tokenizer object to adapt tokenize method such that it returns
    a tuple of nlgmetricverse.Collator object instead of list.
    """
    def __init__(self, tokenizer):
        """
        :param tokenizer: Tokenizer object that implements `tokenize` method.
        """
        self.tokenizer = tokenizer

    def tokenize(self, predictions: Collator, references: Collator) -> Tuple[Collator, Collator]:
        """
        Returns a tuple of nlgmetricverse.Collator object instead of list.

        :param predictions: Predictions.
        :param references: References.
        :return: A tuple of nlgmetricverse.Collator.
        """
        _predictions = []
        _references = []
        for preds in predictions:
            _predictions.append([self.tokenizer.tokenize(pred) for pred in preds])
        for refs in references:
            _references.append([self.tokenizer.tokenize(ref) for ref in refs])
        return Collator(_predictions, keep=True), Collator(_references, keep=True)
