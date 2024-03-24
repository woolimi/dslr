import pandas as pd

def get_valid_values(values: pd.Series) -> pd.Series:
    """
    Return only valid values
    """
    return values[values == values]