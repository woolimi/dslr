import argparse
import os
from lib.core import describe

def csv_file_type(filename):
    if not filename.endswith('.csv') or not os.path.isfile(filename):
        raise argparse.ArgumentTypeError("File '{}' is not a valid .csv file.".format(filename))
    return filename

def main():
    parser = argparse.ArgumentParser(description="describe: a simple script to describe csv dataset")
    parser.add_argument("csv_file", type=csv_file_type, help="Path to the .csv file")
    args = parser.parse_args()
    describe(args.csv_file)

if __name__ == "__main__":
    main()