import argparse
import os
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
    v.draw_histograms()
    # v.distribution_by_house('Arithmancy')
    dist1, dist2 = v.save_distributions()
    print(dist1)
    print(dist2)

if __name__ == "__main__":
    main()