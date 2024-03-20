import argparse
import os
from lib.core import draw_histogram
from lib.core import get_dataframe
from lib.core import get_numeric_columns

def csv_file_type(filename):
    if not filename.endswith('.csv') or not os.path.isfile(filename):
        raise argparse.ArgumentTypeError("File '{}' is not a valid .csv file.".format(filename))
    return filename

def main():
    parser = argparse.ArgumentParser(description="describe: a simple script to describe csv dataset")
    parser.add_argument("csv_file", type=csv_file_type, help="Path to the .csv file")
    args = parser.parse_args()
   
    df = get_dataframe(args.csv_file)
    cols = get_numeric_columns(df)
    draw_histogram(df, cols)

if __name__ == "__main__":
    main()