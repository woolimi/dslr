import argparse
from lib.visualize import Visualization
from lib.validate import is_csv
from lib.print import danger

def main():
    parser = argparse.ArgumentParser(description="draw scatter plots and provide pearson correlation coefficient")
    parser.add_argument("csv_file", type=is_csv, help="Path to the .csv file")
    parser.add_argument("-a", "--all", action="store_true", help="option to show all scatter plots")
    parser.add_argument("columns", nargs="*", help="name columns you want to see")
    args = parser.parse_args()

    if args.all and args.columns:
        parser.error("Please specify either option -a/-all or column name/names, not both")

    v = Visualization(args.csv_file)
    if args.all:
        v.draw_scatter_plots()
    elif args.columns:
        valid_columns = v.get_continuous_columns()
        for i in args.columns:
            if not i in valid_columns:
                print(danger('Error: invalid column name is included'))
                exit(1)
        v.draw_scatter_plot(args.columns)
    else:
        v.get_scatter_plot_answer()

if __name__ == "__main__":
    main()