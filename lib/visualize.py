import pandas as pd #
import matplotlib.pyplot as plt
import seaborn as sns
from .core import get_valid_values
from itertools import combinations
from .math import math_mean, math_count, math_std

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
# pair plot -> coef 기준으로 정렬?
class Visualization:
    criteria = "Hogwarts House"
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    df = None
    cols = None

    @classmethod
    def set_dataframe(cls, filename):
        cls.df = pd.read_csv(filename, index_col="Index")
    
    # couldn't drop NaN...
    @classmethod
    def preprocess_data(cls):
        cls.df['Best Hand'] = cls.df['Best Hand'].map({'Left': 0, 'Right': 1})
        cls.df['Birth Month'] = pd.to_datetime(cls.df['Birthday']).dt.month
    
    @classmethod
    def set_numeric_columns(cls):
        cls.cols = cls.df.select_dtypes('number').columns

    def get_continuous_columns(self):
        return self.df.select_dtypes('float64').columns

    def __init__(self):
        pass

    # TODO: add legend / change x label to 'Best Hand'
    def draw_histograms(self):
        fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=100)
        fig.suptitle('Histogram', fontsize=25)
        # fig.legend(self.houses, loc='lower right', bbox_to_anchor=(1, -0.1))
        ax = None
        for i in range(4):
            for j, column in enumerate(self.cols[i*4:i*4+4]):
                if i == 3 and j == 2:
                    legend_shown = True
                else:
                    legend_shown = False
                ax = sns.histplot(ax=axes[i, j], data=self.df, x=column, hue="Hogwarts House", hue_order=self.houses, legend=legend_shown)
                if j:
                    axes[i, j].set(ylabel='')
        axes[i, j+1].axis('off')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.15, 0.75))
        # plt.show()
        plt.savefig('result.png')

    # def calculate_pearson_corr_coef(column1, column2):


    def draw_scatter_plots(self):
        # prerequisite of pearson coef - it should be continuous / follow bivariate normal distribution...
        cols =self.get_continuous_columns()
        combi = list(combinations(cols, 2))
        fig, axes = plt.subplots(10, 8, figsize=(56, 70))
        fig.suptitle('Scatter plot', fontsize=25)
        ax = None
        print(len(combi))
        for i in range(10):
            for j, (x, y) in enumerate(combi[i * 8: i * 8 + 8]):
                if i == 9 and j == 5:
                    legend_shown = True
                else:
                    legend_shown = False
                ax = sns.scatterplot(ax=axes[i, j], data=self.df, x=x, y=y, hue="Hogwarts House", hue_order=self.houses, legend=legend_shown)
        for f in range(j, 7):
            axes[i, f+1].axis('off')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.15, 0.75))
        plt.savefig('result3.png')

    # is extended or not
    def draw_pair_plot(self, is_expended=False):
        # cf. pair grid : https://seaborn.pydata.org/tutorial/axis_grids.html
        if is_expended is True:
            df = self.df
        else:
            df = self.df.drop(columns=['Best Hand', 'Birth Month'])
        sns.pairplot(data=df, hue="Hogwarts House")
        plt.savefig('result.png')
        #plt.show()

    # 전체 distribution으로부터 얼마나 동떨어져있는지
    def distribution_by_house_from_total_avg(self, column):
        res = {}
        df = self.df[[self.criteria, column]].dropna()
        total_avg = math_mean(df[column])
        for house in self.houses:
            byhouse = df.loc[df[self.criteria] == house, column]
            c = math_count(byhouse)
            squared_diff_sum = 0
            for v in byhouse:
                squared_diff_sum += (v - total_avg) ** 2
            res[house] = squared_diff_sum / (c - 1)
        return res

    # 한 하우스 내에서의 분포가 얼마나 분산되어 있는지
    def distribution_by_house_from_house_avg(self, column):
        res = []
        df = self.df[[self.criteria, column]].dropna()  # 빈 데이터 가공하는 로직 통일하기
        for house in self.houses:
            byhouse = df.loc[df[self.criteria] == house, column]
            c = math_count(byhouse)
            m = math_mean(byhouse)
            squared_diff_sum = 0
            for v in byhouse:
                squared_diff_sum += (v - m) ** 2
            res.append(squared_diff_sum / (c - 1))
        return res

    def save_distributions(self):
        df = pd.DataFrame(index=self.houses)
        df2 = pd.DataFrame(index=self.houses)
        pd.set_option('display.max_columns', None)
        for column in self.cols:
            df[column] =  self.distribution_by_house_from_house_avg(column)
            df2[column] =  self.distribution_by_house_from_total_avg(column)

        # min max or top bottom : add range or deviation -> should i fix index?
        columns = df
        return (df, df2)
