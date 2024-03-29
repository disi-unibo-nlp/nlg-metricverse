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

"""
Custom iterable data structure.
It allows the user to work smoothly with arrays of strings like NLG predictions and references
(with 1:1, 1:N, and N:N scenarios, where N means non-collapsable).
"""
from typing import List, Union

import numpy as np

from nlgmetricverse.utils.data_structure import NestedSingleType


class Collator(list):
    """
    Class inherited from list, with extended functionalities.
    It supports reshaping and several data loading modes, including an automatic conversion from string and (nested) lists.
    """
    def __init__(self, sequence, keep=False):
        sequence = self._construct(sequence, keep=keep)
        super().__init__(sequence)

    @property
    def shape(self):
        return np.array(self, dtype=object).shape

    @property
    def ndim(self):
        return len(self.shape)

    def collapse(self):
        if NestedSingleType.get_type(self, 1) != "list":
            return Collator([item for item in self], keep=True)
        return Collator([item for items in self for item in items], keep=True)

    def nested(self):
        return Collator(self.from_list(self))

    def reshape(self, *args):
        _seq = np.array(self, dtype=object)
        if _seq.shape[:2] == (1, 1):
            return Collator(_seq.ravel().reshape(1, -1).tolist(), keep=True)
        elif _seq.ndim == 3 and _seq.shape[1] == 1:
            args = tuple(list(args) + [-1])
        return Collator(_seq.reshape(args).tolist(), keep=True)

    def reshape_len(self, *args):
        _len = len(self)
        return self.reshape(_len, *args)

    def can_collapse(self):
        if self.ndim >= 2:
            return self.shape[1] == 1
        if isinstance(self[0], list):
            n_item = len(self[0])
            return all([len(items) == n_item for items in self])
        return True

    def to_list(self, collapse=True):
        if collapse:
            return list(self.collapse())
        return list(self)

    def _construct(self, sequence: Union[str, List[str], List[List[str]]], keep: bool) -> List[List[str]]:
        if keep:
            return sequence

        _type_primary = NestedSingleType.get_type(sequence, order=0)
        _type_secondary = NestedSingleType.get_type(sequence, order=1)
        if _type_primary in ["str", "dict"]:
            sequence = self.from_str(sequence)
        elif _type_primary == "list" and _type_secondary != "list":
            sequence = self.from_list(sequence)

        return sequence

    @staticmethod
    def from_list(seq: List[str]):
        return [[item] for item in seq]

    @classmethod
    def from_str(cls, seq: str):
        return [seq]
