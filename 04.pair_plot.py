import argparse
from lib.visualize import Visualization
from lib.validate import is_csv


def main():
    parser = argparse.ArgumentParser(description="describe: a simple script to describe csv dataset")
    parser.add_argument("csv_file", type=is_csv, help="Path to the .csv file")
    args = parser.parse_args()
    v = Visualization()

    v.set_dataframe(args.csv_file)
    v.preprocess_data()
    v.set_numeric_columns()
    v.draw_pair_plot()

if __name__ == "__main__":
    main()