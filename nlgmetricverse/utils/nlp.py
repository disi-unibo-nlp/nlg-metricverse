"""
Utils file for Natural Language Processing
"""
import re
import string


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
