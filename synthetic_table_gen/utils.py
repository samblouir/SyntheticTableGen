# utils.py
# -----------------------------------------------------------------------------
# Utility functions for missing-data logic, random property value generation,
# and random property unit selection.
# -----------------------------------------------------------------------------


import random
import numpy as np

from synthetic_table_gen.data_lists import UNITS, PROPERTY_UNIT_MAP, PROPERTY_RANGES

def maybe_missing_str(base_str: str, p_missing=0.1) -> str:
    """
    With probability p_missing, return 'not available'; else return base_str.

    :param base_str: The original string.
    :param p_missing: Probability that the string is replaced.
    :return: base_str or 'not available'.
    """
    if random.random() < p_missing:
        return "not available"
    return base_str


def maybe_missing_float(rng: np.random.Generator, low=0.0, high=100.0, p_missing=0.1):
    """
    Return a random float within [low, high], or None with probability p_missing.

    :param rng: Numpy Random Generator.
    :param low: Min float.
    :param high: Max float.
    :param p_missing: Probability of returning None.
    :return: A float or None.
    """
    if rng.random() < p_missing:
        return None
    return round(rng.uniform(low, high), 3)


def get_random_property_value(prop_name: str, rng: np.random.Generator):
    """
    Return a random float for a given property using known or default ranges.

    :param prop_name: The property name (canonical form recommended).
    :param rng: Numpy Random Generator.
    :return: A float in the appropriate range, truncated to 3 decimals.
    """
    if prop_name in PROPERTY_RANGES:
        low, high = PROPERTY_RANGES[prop_name]
    else:
        low, high = (0.0, 1e6)
    value = rng.uniform(low, high)
    return round(value, 3)


def get_random_property_unit(prop_name: str, rng: np.random.Generator):
    """
    Return a random unit for a given property if listed in PROPERTY_UNIT_MAP,
    else pick from the entire UNITS list.

    :param prop_name: The property name (canonical form recommended).
    :param rng: Numpy Random Generator.
    :return: A unit string.
    """
    if prop_name in PROPERTY_UNIT_MAP:
        return rng.choice(PROPERTY_UNIT_MAP[prop_name])
    else:
        return rng.choice(UNITS)

