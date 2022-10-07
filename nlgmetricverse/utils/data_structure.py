from typing import Dict, List, Any, Optional


def bulk_remove_keys(obj: Dict, keys: List[str]) -> Dict:
    """
    Remove keys from Dict.

    :param obj: Dict.
    :param keys: Keys to be removed.
    :return: Dict updated.
    """
    return {k: v for k, v in obj.items() if k not in keys}


def get_common_keys(d1: Dict, d2: Dict) -> List[str]:
    """
    Get common keys between two Dicts.

    :param d1: Dict 1.
    :param d2:  Dict 2.
    :return: List of common keys.
    """
    set1 = set(d1.keys())
    set2 = set(d2.keys())

    return list(set1.intersection(set2))


def pop_item_from_dict(d: Dict[str, Any], key: str, default: Any = None, must_exists: bool = False):
    """
    Pop key from Dict d if key exists, return d otherwise.

    :param d: Dictionary for key to be removed.
    :param key: Key name to remove from dictionary d.
    :param default: Default value to return if key not found.
    :param must_exists: Raises an exception if True when given key does not exist.
    :return: Popped value for key if found, None otherwise.
    """
    if key not in d and must_exists:
        raise KeyError(f"'{key}' not found in '{d}'.")
    val = d.pop(key) if key in d else default
    return val


def replace(a: List, obj: object, index=-1):
    """
    Replace an object in a list with another object.

    :param a: List containing the object.
    :param obj: Replacing object.
    :param index: Index of the object to be replaced.
    :return: Updated list.
    """
    del a[index]
    a.insert(index, obj)
    return a


def remove_duplicates(lst):
    return [t for t in (set(tuple(i) for i in lst))]


class NestedSingleType:
    @staticmethod
    def is_iterable(obj):
        """
        Check if an object is iterable.

        :param obj: The object to be checked.
        :return: True if object is iterable, False otherwise.
        """
        if isinstance(obj, str) or isinstance(obj, dict):
            return False
        try:
            iter(obj)
        except TypeError:
            return False
        return True

    @staticmethod
    def join(types: List[str]):
        """
        Join nested types.

        :param types: Types to be nested.
        :return: Joined nested types (lowercase).
        """
        nested_types = f"{types.pop(-1)}"

        for _type in types:
            nested_types = f"{_type}<{nested_types}>"
        return nested_types.lower()

    @classmethod
    def get_type(cls, obj, order: Optional[int] = None):
        """
        Get object type.

        :param obj: Object inspected.
        :param order: Optional, if there is a preferred index (e.g., 0=primary, 1=secondary).
        :return: Object type.
        """
        _obj = obj

        types = []
        while cls.is_iterable(_obj):
            types.append(type(_obj).__name__)
            _obj = _obj[0]
        types.append(type(_obj).__name__)

        if order is not None:
            try:
                return types[order]
            except IndexError:
                return None

        return cls.join(types)
