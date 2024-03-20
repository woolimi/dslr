import argparse
import pandas as pd
from lib.core import get_valid_values
from lib.validate import is_csv
from lib.math import math_count, math_unique, math_top, math_freq,\
    math_mean, math_min, math_max, math_std, math_quartiles

def describe(filename: str):
    """
    Describe the dataset in filename.
    """
    df = pd.read_csv(filename, index_col="Index")

    df['Birthday'] = pd.to_datetime(df['Birthday'])
    df.drop(labels=['First Name', 'Last Name'], axis='columns', inplace=True)

    # To compare with pandas describe function
    # print(df.describe(include='all'))

    summary = pd.DataFrame()    
    summary.index = ['count', 'unique', 'top', 'freq', 'mean', 'min', '25%', '50%', '75%', 'max', 'std']

    for column in df.columns:
        values = get_valid_values(df[column])
        summary[column] = [
            math_count(values),
            math_unique(values),
            math_top(values),
            math_freq(values),
            math_mean(values),
            math_min(values),
            math_quartiles(values, 0.25),
            math_quartiles(values, 0.5),
            math_quartiles(values, 0.75),
            math_max(values),
            math_std(values),
        ]

    print(summary)
    return summary


def main():
    """
    Main function to parse arguments and call describe function
    """
    parser = argparse.ArgumentParser(description="describe: a simple script to describe csv dataset")
    parser.add_argument("csv_file", type=is_csv, help="Path to the .csv file")
    args = parser.parse_args()
    describe(args.csv_file)

if __name__ == "__main__":
    main()