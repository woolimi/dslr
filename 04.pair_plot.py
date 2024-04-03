import argparse
from lib.visualize import Visualization
from lib.validate import is_csv


def main():
    parser = argparse.ArgumentParser(description="describe: a simple script to describe csv dataset")
    parser.add_argument("csv_file", type=is_csv, help="Path to the .csv file")
    parser.add_argument("-a", "--all", action="store_true", help="option to include non-continuous columns")
    args = parser.parse_args()
    v = Visualization(args.csv_file)
    v.draw_pair_plot(args.all)

if __name__ == "__main__":
    main()