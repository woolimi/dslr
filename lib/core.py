import pandas as pd
import numpy as np

def describe(filename):
    df = pd.read_csv(filename, index_col="Index")
