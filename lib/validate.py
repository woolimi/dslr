import argparse
import os
import pandas as pd

def is_csv(filename):
    if not filename.endswith('.csv') or not os.path.isfile(filename):
        raise argparse.ArgumentTypeError("File '{}' is not a valid .csv file.".format(filename))
    return filename

def is_train_dataset(df: pd.DataFrame):
    if df['Hogwarts House'].isnull().any():
        return False
    return True
