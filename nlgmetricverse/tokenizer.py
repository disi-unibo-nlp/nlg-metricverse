from typing import List, Tuple

from nlgmetricverse.collator import Collator
from nlgmetricverse.utils.nlp import normalize_text


class DefaultTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return normalize_text(text).split()


class TokenizerWrapper:
    """
    Wraps the tokenizer object to adapt tokenize method such that it returns
    a tuple of nlgmetricverse.Collator object instead of list.
    Args:
        tokenizer: Tokenizer object that implements `tokenize` method.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, predictions: Collator, references: Collator) -> Tuple[Collator, Collator]:
        _predictions = []
        _references = []
        for preds in predictions:
            _predictions.append([self.tokenizer.tokenize(pred) for pred in preds])
        for refs in references:
            _references.append([self.tokenizer.tokenize(ref) for ref in refs])
        return Collator(_predictions, keep=True), Collator(_references, keep=True)
