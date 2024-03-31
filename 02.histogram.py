import argparse
import os
from lib.visualize import Visualization
from lib.validate import is_csv
from lib.print import danger

# def histogram(filename: str, option:str)
def main():
    parser = argparse.ArgumentParser(description="histogram: draw histogram and show distributions btw attributes and houses")
    parser.add_argument("csv_file", type=is_csv, help="Path to the .csv file")
    parser.add_argument("-a", "--all", action="store_true", help="option to show all histograms")
    parser.add_argument("columns", nargs="*", help="name columns you want to see")
    # parser.add_argument("--aggregate", action="store_true", help="option to show histogram(s) not devided by houses")
    # parser.add_argument("--separate", action="store_true", help="option to show histogram(s) devided by houses (default)") 
    args = parser.parse_args()

    if not args.all and not args.columns:
        parser.error("Please specify either option -a/-all or column name/names")

    v = Visualization(args.csv_file)
    if args.all:
        v.draw_all_histograms() #
    else:
        valid_columns = v.get_numeric_columns()
        for i in args.columns:
            if not i in valid_columns:
                print(danger('Error: invalid column name is included'))
                exit(1)
        v.draw_histogram(args.columns)

if __name__ == "__main__":
    main()