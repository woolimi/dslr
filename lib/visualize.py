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
    
    @classmethod
    def preprocess_data(cls):
        cls.df['Best Hand'] = cls.df['Best Hand'].map({'Left': 0, 'Right': 1})
        cls.df['Birth Month'] = pd.to_datetime(cls.df['Birthday']).dt.month
    
    @classmethod
    def set_numeric_columns(cls):
        cls.cols = cls.df.select_dtypes('number').columns

    def __init__(self):
        pass

    # TODO: add legend / change x label to 'Best Hand'
    def draw_histograms(self):
        fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=100)
        fig.suptitle('Histogram')
        fig.legend(self.houses, loc='lower right', bbox_to_anchor=(1, -0.1))
        for i in range(4):
            for j, column in enumerate(self.cols[i*4:i*4+4]):
                ax = sns.histplot(ax=axes[i, j], data=self.df, x=column, hue="Hogwarts House", hue_order=self.houses, legend=False)
                if j:
                    axes[i, j].set(ylabel='')
        axes[i, j+1].axis('off')
        # legend = fig.add_subplot(4, 4, 16)
        # legend.axis('off')
        # handles, labels = ax.get_legend_handles_labels()
        # axes[3, 3].legend(handle, label)
        # plt.legend(self.houses, loc='center')
        
        # print(self.houses)
        # print(ax.legend_)
        # axes[i, j + 1].legend(labels=self.houses, loc='upper center', title="by Houses")
        # ax.legend(bb)
        plt.tight_layout()
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

    # how to?
        # from total_avg -> calculate variance
        # from each avg -> calculate variance
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
            res[house] = squared_diff_sum
        return res

    # 한 하우스 내에서의 분포가 얼마나 분산되어 있는지
    def distribution_by_house_from_house_avg(self, column):
        res = {}
        df = self.df[[self.criteria, column]].dropna()  # logic 통일할 필요성?ㄴㄴ
        for house in self.houses:
            byhouse = df.loc[df[self.criteria] == house, column]
            c = math_count(byhouse)
            m = math_mean(byhouse)
            squared_diff_sum = 0
            for v in byhouse:
                squared_diff_sum += (v - m) ** 2
            res[house] = squared_diff_sum / c
        return res

    def save_distributions(self):
        df = pd.DataFrame(self.houses, columns=['Houses'])
        pd.set_option('display.max_columns', None)
        for column in self.cols:
            # df[column] =  self.distribution_by_house_from_house_avg(column)
            # df = df.append(column, distribution_by_house_from_total_avg(self, column), ignore_index=True) -> append deprecated
            df[column] = df['Houses'].map(self.distribution_by_house_from_house_avg(column))    # sorting 안하면 그냥 column으로 바로 추가하는게 나을지도?
        return (df)