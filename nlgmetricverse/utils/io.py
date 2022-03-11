"""
Utils functions for I/O operations.
"""
import json
import os
import pickle
from typing import Any, Dict, Union


def json_load(fp: str) -> Union[Dict, None]:
    """
    Try loading json and return parsed dictionary. Returns None if file does not exist, or in case of a serialization
    error.

    :param fp: File path.
    :return: Parsed dictionary, None if file does not exist.
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
    Saves a dictionary as json file to given fp.

    :param obj: Dict to be saved.
    :param fp: File path.
    :param overwrite: bool.
    :return: None.
    """
    if os.path.exists(fp) and not overwrite:
        raise ValueError(f"Path {fp} already exists. To overwrite, use `overwrite=True`.")

    with open(fp, "w") as jf:
        json.dump(obj, jf)


def pickle_load(fp: str) -> Any:
    """
    Try loading a dictionary as json file.

    :param fp: File path.
    :return: Parsed dictionary.
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
    Saves a dictionary as json file to given fp.

    :param obj: Dict to be saved.
    :param fp: File path.
    :param overwrite: bool.
    :return: None.
    """
    if os.path.exists(fp) and not overwrite:
        raise ValueError(f"Path {fp} already exists. To overwrite, use overwrite=True.")

    with open(fp, "wb") as pkl:
        pickle.dump(obj, pkl)
