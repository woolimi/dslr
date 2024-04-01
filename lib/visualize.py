import pandas as pd #
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from .math import math_mean, math_count, math_std, math_max, math_min
from math import sqrt
from .validate import is_train_dataset
from typing import List
from lib.print import success, danger
from pandas.api.types import is_numeric_dtype

class Visualization:
    criteria = "Hogwarts House"
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']

    def set_dataframe(self, filename: str):
        self.df = pd.read_csv(filename, index_col="Index")
    
    def preprocess_data(self):
        if not self.is_houses:
            self.df = self.df.drop([self.criteria], axis=1)
        self.df['Best Hand'] = self.df['Best Hand'].map({'Left': 0, 'Right': 1})
        self.df['Birth Month'] = pd.to_datetime(self.df['Birthday']).dt.month
    
    def set_numeric_columns(self):
        self.cols = self.df.select_dtypes('number').columns
    
    def get_numeric_columns(self)->pd.DataFrame:
        return self.cols

    def get_continuous_columns(self):
        return self.df.select_dtypes('float64').columns
    
    def __init__(self, filename: str):
        self.set_dataframe(filename)
        self.is_houses = is_train_dataset(self.df) # and not is_aggregate
        self.preprocess_data()
        self.set_numeric_columns()
        
    def distribution_by_house_from_total_avg(self, column, is_normalize=True):
        res = []
        df = self.df[[self.criteria, column]].dropna()

        # normalize
        if is_normalize:
            min = math_min(df[column])
            max = math_max(df[column])
            df[column] = df[column].apply(lambda x: (x - min) / (max - min))

        total_avg = math_mean(df[column])
        for house in self.houses:
            byhouse = df.loc[df[self.criteria] == house, column]
            c = math_count(byhouse)
            squared_diff_sum = 0
            for v in byhouse:
                squared_diff_sum += (v - total_avg) ** 2
            res.append(squared_diff_sum / (c - 1))
        return res

    def distribution_by_house_from_house_avg(self, column, is_normalize=True):
        res = []
        df = self.df[[self.criteria, column]].dropna()

        for house in self.houses:
            byhouse = df.loc[df[self.criteria] == house, column]
            
            if is_normalize:
                min = math_min(byhouse)
                max = math_max(byhouse)
                byhouse = byhouse.apply(lambda x: (x - min) / (max - min))
            
            c = math_count(byhouse)
            m = math_mean(byhouse)
            squared_diff_sum = 0
            for v in byhouse:
                squared_diff_sum += (v - m) ** 2
            res.append(squared_diff_sum / (c - 1))
        return res

    def _get_variance(self, column: pd.Series, is_normalize=True)->float:
        if not is_numeric_dtype(column.dtype):
            return float('nan')
        col = column.dropna()
        if is_normalize:
            min = math_min(col)
            max = math_max(col)
            col = col.apply(lambda x: (x - min) / (max - min))
            return (math_std(col) ** 2)
        return math_std(col) ** 2

    def _cal_distributions(self):
        if self.is_houses:
            # generate dataframes using house names as index
            df = pd.DataFrame(index=self.houses)
            df2 = pd.DataFrame(index=self.houses)
            pd.set_option('display.max_columns', None)
            # write figures
            for column in self.cols:
                df[column] =  self.distribution_by_house_from_total_avg(column)
                df2[column] =  self.distribution_by_house_from_total_avg(column, False)
            # add range row
            ranges = df.apply(lambda col: math_max(col)-math_min(col))
            ranges2 = df2.apply(lambda col: math_max(col)-math_min(col))
            df.loc['Range'] = ranges
            df2.loc['Range'] = ranges2
            # sort by range
            df = df.sort_values(by='Range', ascending=True, axis=1)
            df2 = df2.sort_values(by='Range', ascending=True, axis=1)
            return (df, df2)
        else:
            # variance by attributes
            var = pd.DataFrame()
            var_normalized = pd.DataFrame()
            df = self.df.select_dtypes(include='number')
            mins = df.apply(math_min)
            maxs = df.apply(math_max)
            var['variance'] = df.apply(self._get_variance, args=(False,))
            var_normalized['variance'] = df.apply(self._get_variance, args=(True,))
            var = var.sort_values(by='variance', ascending=True)
            var_normalized = var_normalized.sort_values(by='variance', ascending=True)
            return (var_normalized['variance'], var['variance'])

    def print_distributions(self):
        (d1, d2) = self._cal_distributions()
        print(d1)
        print('------------------- before normalization ---------------------')
        print(d2)

    def draw_all_histograms(self):
        fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=100)
        fig.suptitle('Histogram', fontsize=25)
        for i in range(4):
            for j, column in enumerate(self.cols[i*4:i*4+4]):
                if self.is_houses:
                    legend_shown = i == 3 and j == 2
                    ax = sns.histplot(ax=axes[i, j], data=self.df, x=column, hue=self.criteria, hue_order=self.houses, legend=legend_shown)
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
        for c in columns:
            if self.is_houses:
                ax = sns.histplot(data=self.df, x=c, hue=self.criteria, hue_order=self.houses, legend=True)
                vars = self.distribution_by_house_from_total_avg(c, False)
                text = f"[ {c} ]\n"
                for i in range(4):
                    text = text + f"{self.houses[i][0]} : {'{:.4f}'.format(vars[i])}\n"
                print(text)
            else:
                ax = sns.histplot(data=self.df, x=c)
                text = f"variance : {math_std(self.df[c]) ** 2}"
                x_pos = ax.get_xlim()[0]
                y_pos = ax.get_ylim()[1]
                ax.text(x_pos, y_pos, text)
            plt.show()
            
    def _calculate_corr_coef(self, column1:str, column2:str):
        df = self.df[[column1, column2]]
        means = df.apply(math_mean)
        np = df.fillna(means).to_numpy()
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

    