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
Utils functions for I/O operations:
- read/write json files
- read/write pickle files
"""
import json
import os
import pickle
from typing import Any, Dict, Union


def json_load(fp: str) -> Union[Dict, None]:
    """
    Load a json file and return the parsed dictionary.
    Return None if file does not exist, or in case of a serialization error.

    :param fp: Json file path.
    :return: Parsed dictionary, None if the file does not exist.
    """
    try:
        with open(fp, "r") as jf:
            _json = json.load(jf)
    except FileNotFoundError:
        return
    except json.JSONDecodeError:
        return
    else:
        return _json


def json_save(obj: Dict, fp: str, overwrite: bool = True) -> None:
    """
    Save a dictionary as a json file to the specified file path.

    :param obj: Dict to be saved.
    :param fp: File path.
    :param overwrite: bool.
    """
    if os.path.exists(fp) and not overwrite:
        raise ValueError(f"Path {fp} already exists. To overwrite, use `overwrite=True`.")

    with open(fp, "w") as jf:
        json.dump(obj, jf)


def pickle_load(fp: str) -> Any:
    """
    Load a pickle file (serialized Python object).

    :param fp: File path.
    :return: Parsed Python object.
    """
    try:
        with open(fp, "rb") as pkl:
            _obj = pickle.load(pkl)
    except FileNotFoundError:
        return
    else:
        return _obj


def pickle_save(obj: Dict, fp: str, overwrite: bool = True) -> None:
    """
    Serialize a dictionary object with pickle to the specified file path.

    :param obj: Dict to be saved.
    :param fp: File path.
    :param overwrite: bool.
    """
    if os.path.exists(fp) and not overwrite:
        raise ValueError(f"Path {fp} already exists. To overwrite, use overwrite=True.")

    with open(fp, "wb") as pkl:
        pickle.dump(obj, pkl)
