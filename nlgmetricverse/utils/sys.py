import logging
import os
import sys


def set_env(name: str, value: str):
    """
    Set a Py environment.

    :param name: Key of the environment.
    :param value: Value of the environment.
    :return: Any.
    """
    if not isinstance(value, str):
        raise ValueError(f"Expected type str for 'value', got {type(value)}.")
    os.environ[name] = value


def log(message):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(message)
