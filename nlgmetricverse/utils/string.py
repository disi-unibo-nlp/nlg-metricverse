import re
import string


def camel_to_snake(name):
    """
    Convert a string from camel case to snake case.

    :param name: The string to be converted.
    :return: The converted string.
    """
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def normalize_text(text: str, uncased: bool = True) -> str:
    """
    Attempt to reduce a text randomness, bringing it closer to a predefined “standard”.

    :param text: String to be normalized.
    :param uncased: bool, if true format string to lower case.
    :return: Normalized string.
    """
    pattern = r"[%s]" % re.escape(string.punctuation)
    text = re.sub(pattern, " ", text)
    normalized_text = " ".join(text.split())
    if uncased:
        normalized_text = normalized_text.lower()
    return normalized_text
