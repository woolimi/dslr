import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_valid_values(values: pd.Series) -> pd.Series:
    """
    Return only valid values
    """
    return values[values == values]


def describe(filename):
    df = pd.read_csv(filename, index_col="Index")

# def get_columns(
# draw_histogram
    # which Hogwarts course has a homogenous score "distribution" between all four houses
# draw_scatter_plot
    # what are the two features that are similar
# draw_pair_plot
    # pair plot or scatter plot matrix

    # from the visualization, what features are you going to use for your logistic regression?

# Hogwarts House <- constantize?
def get_dataframe(filename):
    df = pd.read_csv(filename, index_col="Index")
    return df

def get_numeric_columns(dataframe):
    return dataframe.select_dtypes('number').columns

def draw_histogram(df, cols):
    for column in cols:
        sns.histplot(data=df, x=column, hue="Hogwarts House")
        plt.show()
