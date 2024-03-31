import pandas as pd #
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from .math import math_mean, math_count, math_std, math_max, math_min
from math import sqrt
from .validate import is_train_dataset
from typing import List
from lib.print import success, danger


# def get_columns(
# draw_histogram
    # which Hogwarts course has a homogenous score "distribution" between all four houses
# draw_scatter_plot
    # what are the two features that are similar
# draw_pair_plot
    # from the visualization, what features are you going to use for your logistic regression?

# parsing
# draw
# histogram -> avg / variance btw houses --> 정렬?
# scatterplot -> coef --> 정렬?
# pair plot -> coef 기준으로 정렬?
class Visualization:
    criteria = "Hogwarts House"
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    is_houses = None
    df = None
    cols = None

    @classmethod
    def set_dataframe(cls, filename: str):
        cls.df = pd.read_csv(filename, index_col="Index")
    
    @classmethod
    def preprocess_data(cls):
        print(cls.df)
        if not cls.is_houses:
            cls.df.drop([cls.criteria], axis=1, inplace=True)
        cls.df['Best Hand'] = cls.df['Best Hand'].map({'Left': 0, 'Right': 1})
        cls.df['Birth Month'] = pd.to_datetime(cls.df['Birthday']).dt.month
        print(cls.df)
    
    @classmethod
    def set_numeric_columns(cls):
        cls.cols = cls.df.select_dtypes('number').columns
    
    def get_numeric_columns(cls)->pd.DataFrame:
        return cls.cols

    def get_continuous_columns(self):
        return self.df.select_dtypes('float64').columns
    
    # TODO : is_Aggregate
    def __init__(self, filename: str):
        self.set_dataframe(filename)
        self.is_houses = is_train_dataset(self.df) # and not is_aggregate
        self.preprocess_data()
        self.set_numeric_columns()
        

    def _normalization(df: pd.DataFrame):
        min = df.apply(math_min)
        max = df.apply(math_max)
        normalized_df = (df - min) / (max - min)
        # df[:] =  normalized_df
        return (normalized_df)

    # TODO: add legend / change x label to 'Best Hand'
    def draw_all_histograms(self):
        fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=100)
        fig.suptitle('Histogram', fontsize=25)
        ax = None
        for i in range(4):
            for j, column in enumerate(self.cols[i*4:i*4+4]):
                if self.is_houses:
                    legend_shown = i == 3 and j == 2
                    ax = sns.histplot(ax=axes[i, j], data=self.df, x=column, hue=self.criteria, hue_order=self.houses, multiple='dodge', legend=legend_shown)
                else:
                    ax = sns.histplot(ax=axes[i, j], data=self.df, x=column)
                if j:
                    axes[i, j].set(ylabel='')
        axes[i, j+1].axis('off')
        if self.is_houses:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.15, 0.75))
        plt.savefig('histograms.png')
        #plt.show()
        print(success("Histograms are saved in histograms.png"))

    def draw_histogram(self, columns: List[str]):
        print(self.df)
        for c in columns:
            if self.is_houses:
                ax = sns.histplot(data=self.df, x=c, hue=self.criteria, hue_order=self.houses, legend=True)
            else:
                ax = sns.histplot(data=self.df, x=c)
            var = math_std(self.df[i]) ** 2
            text = "variance : " + str(var) + "\n"
            if self.is_houses:
                text += "btw houses by house avg : " + str(self.distribution_by_house_from_house_avg(c)) + "\n"
                text += "btw houses by total avg : " + str(self.distribution_by_house_from_total_avg(c))
            # ax.text(2, 2, text, fontsize=10)
            print(text)
            plt.show()
            
    def _calculate_corr_coef(self, column1:str, column2:str):
        df = self.df[[column1, column2]]
        means = df.apply(math_mean)
        np = df.fillna(means).to_numpy()  # fill NaN with mean <
        means = means.to_numpy()
        co_var = 0
        sq_dev_x = 0
        sq_dev_y = 0
        np_len = len(np)
        for i in range (np_len):
            dev = np[i] - means
            co_var += dev[0] * dev[1]
            sq_dev_x += dev[0] * dev[0]
            sq_dev_y += dev[1] * dev[1]
        coef = co_var / (sqrt(sq_dev_x) * sqrt(sq_dev_y))
        return coef

    def draw_scatter_plots(self):
        # prerequisite of pearson coef - it should be continuous / follow bivariate normal distribution...
        cols =self.get_continuous_columns()
        combi = list(combinations(cols, 2))
        fig, axes = plt.subplots(10, 8, figsize=(56, 70))
        fig.suptitle('Scatter plot', fontsize=25)
        for i in range(10):
            for j, (x, y) in enumerate(combi[i * 8: i * 8 + 8]):
                self._calculate_corr_coef(x, y)
                legend_shown = self.is_houses and i == 9 and j == 5
                if self.is_houses:
                    ax = sns.scatterplot(ax=axes[i, j], data=self.df, x=x, y=y, \
                        hue="Hogwarts House", hue_order=self.houses, legend=legend_shown)
                else:
                    ax = sns.scatterplot(ax=axes[i, j], data=self.df, x=x, y=y, legend=legend_shown)
                x_pos = ax.get_xlim()[0]
                y_pos = ax.get_ylim()[1]
                ax.text(x_pos, y_pos, 'PCC : ' + str(self._calculate_corr_coef(x, y)))
        for f in range(j, 7):
            axes[i, f+1].axis('off')
        if self.is_houses:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.15, 0.75))
        plt.savefig('scatter_plots.png')
        print(success("Scatter plots are saved in scatter_plots.png"))

    def draw_scatter_plot(self, columns: List[str]):
        if len(columns) < 2:
            columns = columns * 2
        combi = list(combinations(columns, 2))
        if self.is_houses:
            for c1, c2 in combi:
                ax = sns.scatterplot(data=self.df, x=c1, y=c2, \
                    hue="Hogwarts House", hue_order=self.houses, legend=True)
                x_pos = ax.get_xlim()[0]
                y_pos = ax.get_ylim()[1]
                ax.text(x_pos, y_pos, 'PCC : ' + str(self._calculate_corr_coef(c1, c2)))
                plt.show()
        else:
            for c1, c2 in combi:
                ax = sns.scatterplot(data=self.df, x=c1, y=c2)
                x_pos = ax.get_xlim()[0]
                y_pos = ax.get_ylim()[1]
                ax.text(x_pos, y_pos, 'PCC : ' + str(self._calculate_corr_coef(c1, c2)))
                plt.show()

    def draw_pair_plot(self, is_expended=False):
        if is_expended is True:
            df = self.df
        else:
            df = self.df.drop(columns=['Best Hand', 'Birth Month'])
        if self.is_houses:
            sns.pairplot(data=df, hue="Hogwarts House")
        else:
            sns.pairplot(data=df)
        plt.savefig('pair_plot.png')
        print(success("Pair plot is saved in pair_plot.png"))

    # 전체 distribution으로부터 얼마나 동떨어져있는지
    def distribution_by_house_from_total_avg(self, column):
        res = []
        df = self.df[[self.criteria, column]].dropna()

        # normalize
        df[column] = self._normalization(df[column])

        total_avg = math_mean(df[column])
        for house in self.houses:
            byhouse = df.loc[df[self.criteria] == house, column]
            c = math_count(byhouse)
            squared_diff_sum = 0
            for v in byhouse:
                squared_diff_sum += (v - total_avg) ** 2
            res.append(squared_diff_sum / (c - 1))
        return res

    # 한 하우스 내에서의 분포가 얼마나 분산되어 있는지 # normalize 필요....?
    def distribution_by_house_from_house_avg(self, column):
        res = []
        df = self.df[[self.criteria, column]].dropna()
        
        # normalize
        df[column] = self._normalization(df[column])
        
        for house in self.houses:
            byhouse = df.loc[df[self.criteria] == house, column]
            c = math_count(byhouse)
            m = math_mean(byhouse)
            squared_diff_sum = 0
            for v in byhouse:
                squared_diff_sum += (v - m) ** 2
            res.append(squared_diff_sum / (c - 1))
        return res

    def get_distributions(self):
        # generate dataframes using house names as index
        df = pd.DataFrame(index=self.houses)
        df2 = pd.DataFrame(index=self.houses)
        pd.set_option('display.max_columns', None)
        # write figures
        for column in self.cols:
            df[column] =  self.distribution_by_house_from_house_avg(column)
            df2[column] =  self.distribution_by_house_from_total_avg(column)
        # add range row
        ranges = df.apply(lambda col: math_max(col)-math_min(col))
        ranges2 = df2.apply(lambda col: math_max(col)-math_min(col))
        df.loc['Range'] = ranges
        df2.loc['Range'] = ranges2
        # sort by range
        df = df.sort_values(by='Range', ascending=True, axis=1)
        df2 = df2.sort_values(by='Range', ascending=True, axis=1)

        # variance by attributes
        df3 = pd.DataFrame()
        df4 = pd.DataFrame()
        df3['variance'] = self.df.select_dtypes(include='number').apply(lambda col: math_std(col)**2)
        df4['variance'] = self.df.select_dtypes(include='number').apply(self._normalization).apply(lambda col: math_std(col)**2)
        # print(df3['Transfiguration'])
        # print(df4['Transfiguration'])
        df3 = df3.sort_values(by='variance', ascending=True)
        df4 = df4.sort_values(by='variance', ascending=True)
        print(df3)
        print(df4)
        return (df, df2, df3)