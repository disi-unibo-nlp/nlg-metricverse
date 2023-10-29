"""
MIT License

Copyright (c) 2021 Open Business Software Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
