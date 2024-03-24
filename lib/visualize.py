import pandas as pd #
import matplotlib.pyplot as plt
import seaborn as sns
from .core import get_valid_values
from itertools import combinations

# def get_columns(
# draw_histogram
    # which Hogwarts course has a homogenous score "distribution" between all four houses
# draw_scatter_plot
    # what are the two features that are similar
# draw_pair_plot
    # pair plot or scatter plot matrix

    # from the visualization, what features are you going to use for your logistic regression?

# parsing
# draw
# histogram -> avg / variance btw houses --> 정렬?
# scatterplot -> coef --> 정렬?
# pair plot -> coef 기준으로 정렬할까?
class Visualization:
    criteria = "Hogwarts House"
    df = None
    cols = None

    @classmethod
    def __set_dataframe__(cls, filename):
        cls.df = pd.read_csv(filename, index_col="Index")
    
    @classmethod
    def __preprocess_data__(cls):
        cls.df['Best Hand'] = cls.df['Best Hand'].map({'Left': 0, 'Right': 1})
        cls.df['Birth Month'] = pd.to_datetime(cls.df['Birthday']).dt.month
    
    @classmethod
    def set_numeric_columns(cls):
        cls.cols = cls.df.select_dtypes('number').columns

    def __init__(self, filename):
        self.__set_dataframe__(filename)
        self.__preprocess_data__()
        self.set_numeric_columns()
        #print(self.df.describe())

    # TODO: add legend / change x label to 'Best Hand'
    def draw_histograms(self):
        fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=100)
        fig.suptitle('Histogram')
        for i in range(4):
            for j, column in enumerate(self.cols[i*4:i*4+4]):
                sns.histplot(ax=axes[i, j], data=self.df, x=column, hue="Hogwarts House", legend=False)
                if j:
                    axes[i, j].set(ylabel='')
        # plt.show()
        plt.savefig('result2.png')

    # def draw_histogram(self, column):
    #     sns.histplot(data=self.df, x=column, hue=self.criteria)
    #     plt.show()

    def draw_scatter_plot(self):
        combi = list(combinations(self.cols, 2))
        for (x, y) in combi:
            sns.scatterplot(data=self.df, x=x, y=y, hue="Hogwarts House")
            plt.show()

    def draw_pair_plot(self):
        # cf. pair grid : https://seaborn.pydata.org/tutorial/axis_grids.html
        sns.pairplot(data=self.df, hue="Hogwarts House")
        plt.savefig('result.png')
        #plt.show()
