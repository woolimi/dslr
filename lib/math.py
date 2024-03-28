import pandas as pd
import numpy as np

def math_count(values: pd.Series) -> int:
    """
    Return the number of values
    """
    return len(values)

def math_unique(values: pd.Series) -> int:
    """
    Return the number of unique values
    """
    dtype = values.dtype
    if dtype == 'datetime64[ns]' or dtype == 'float64':
        return float('nan')

    unique_values = set(values)
    return len(unique_values)

def math_top(values: pd.Series) -> str:
    """
    Return the top value
    """
    dtype = values.dtype
    if dtype == 'datetime64[ns]' or dtype == 'float64':
        return float('nan')

    value_counts = {}
    for value in values:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1
    return max(value_counts, key=value_counts.get)

def math_freq(values: pd.Series) -> int:
    """
    Return the frequency of the top value
    """
    dtype = values.dtype
    if dtype == 'datetime64[ns]' or dtype == 'float64':
        return float('nan')

    value_counts = {}
    for value in values:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1
    return max(value_counts.values())

def math_mean(values: pd.Series) -> float:
    """
    Return the mean of the values
    """
    dtype = values.dtype
    if dtype != 'float64' and dtype != 'datetime64[ns]':
        return float('nan')
    nb_values = math_count(values)
    total = 0
    for value in values:
        if dtype == 'datetime64[ns]':
            tmp = (value - pd.Timestamp(0)).total_seconds()
            tmp = float(tmp)
            total += tmp
        elif value == value:
            total += value
    mean = total / nb_values
    if dtype == 'datetime64[ns]':
        mean = pd.Timestamp(0) + pd.to_timedelta(mean, unit='s')
    return mean


def math_std(values: pd.Series) -> float:
    """
    Return the sample standard deviation of the values
    """
    dtype = values.dtype
    if dtype != 'float64':
        return float('nan')
    m = math_mean(values)
    c = math_count(values)

    sum = 0
    for value in values:
        if value != value:
            value = 0
        sum += (value - m) ** 2
    return (sum / (c - 1)) ** 0.5

def math_min(values: pd.Series) -> float:
    """
    Return the minimum value
    """
    dtype = values.dtype
    if dtype != 'float64' and dtype != 'datetime64[ns]':
        return float('nan')

    minimum = values[0]
    for value in values:
        if value < minimum:
            minimum = value
    return minimum

def math_max(values: pd.Series) -> float:
    """
    Return the maximum value
    """
    dtype = values.dtype
    if dtype != 'float64' and dtype != 'datetime64[ns]':
        return float('nan')

    maximum = values[0]
    for value in values:
        if value > maximum:
            maximum = value
    return maximum



def math_quartiles(values: pd.Series, position: float) -> float:
    """
    Return the quartile at position
    """
    dtype = values.dtype
    if dtype != 'float64' and dtype != 'datetime64[ns]':
        return float('nan')
    sorted_list = sorted(values)
    n = len(sorted_list)
    return _interpolate(sorted_list, (n - 1) * position)

def _interpolate(values, index):
    """
    Interpolate the value at index
    """
    lower_index = int(index)
    upper_index = lower_index + 1
    weight = index - lower_index

    lower_value = values[lower_index]
    upper_value = values[upper_index] if upper_index < len(values) else values[lower_index]

    return lower_value + (upper_value - lower_value) * weight
