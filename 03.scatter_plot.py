import argparse
from lib.core import draw_scatter_plot
from lib.core import get_dataframe
from lib.core import get_numeric_columns
from lib.validate import is_csv


def main():
    parser = argparse.ArgumentParser(description="describe: a simple script to describe csv dataset")
    parser.add_argument("csv_file", type=is_csv, help="Path to the .csv file")
    args = parser.parse_args()
    df = get_dataframe(args.csv_file)
    cols = get_numeric_columns(df)
    draw_scatter_plot(df, cols)

if __name__ == "__main__":
    main()