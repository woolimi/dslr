import argparse
import os

def is_csv(filename):
    if not filename.endswith('.csv') or not os.path.isfile(filename):
        raise argparse.ArgumentTypeError("File '{}' is not a valid .csv file.".format(filename))
    return filename
