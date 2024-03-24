import pandas as pd #
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# def get_columns(
# draw_histogram
    # which Hogwarts course has a homogenous score "distribution" between all four houses
# draw_scatter_plot
    # what are the two features that are similar
# draw_pair_plot
    # pair plot or scatter plot matrix

    # from the visualization, what features are you going to use for your logistic regression?

# Hogwarts House <- constantize?
# classification for visualization?
def get_dataframe(filename):
    df = pd.read_csv(filename, index_col="Index")
    return df

def get_numeric_columns(dataframe):
    return dataframe.select_dtypes('number').columns

# def draw_histogram(df, cols):
#     for column in cols:
#         sns.histplot(data=df, x=column, hue="Hogwarts House")
#         plt.show()

def draw_histogram(df, cols):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=100)
    fig.suptitle('Histogram')
    for i in range(4):
        for j, column in enumerate(cols[i*4:i*4+4]):
            sns.histplot(ax=axes[i, j], data=df, x=column, hue="Hogwarts House", legend=False)
            if j:
                axes[i, j].set(ylabel='')
    # plt.show()
    plt.savefig('result2.png')

def draw_scatter_plot(df, cols):
    combi = list(combinations(cols, 2))
    for (x, y) in combi:
        sns.scatterplot(data=df, x=x, y=y, hue="Hogwarts House")
        plt.show()

def draw_pair_plot(df, cols):
    # cf. pair grid : https://seaborn.pydata.org/tutorial/axis_grids.html
    sns.pairplot(data=df, hue="Hogwarts House")
    plt.savefig('result.png')
    #plt.show()
