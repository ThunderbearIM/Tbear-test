"""
Script to animate the barplot for the file "Cumulative_ListOfDeaths.csv"
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


class BarplotAnimater:

    def __init__(self):
        pass

    def read_csv(self, path: str) -> pd.DataFrame():
        """
        Returns a pandas dataframe from a csv file
        :param path: path to csv file
        :return: pandas dataframe
        """
        df = pd.read_csv(path)
        df = df.dropna()
        df = df.set_index(df.columns[0])
        return df

    def animate(self, i):
        """
        Animates the barplot, instead of showing the Column name as the X-axis, we show the index
        We also rotate the barplot to be horizontal using barh
        We also show the value of each bar using plt.text
        :param i: index
        :return: barplot
        """
        data = self.read_csv("cumsum_list_of_deaths_ulduar.csv").sort_values(by=str(i+1))
        plt.cla()
        plt.barh(data.index, data.iloc[:, i])
        for index, value in enumerate(data.iloc[:, i]):
            plt.text(value, index, str(value))
        plt.xticks(rotation=90)
        plt.xlabel("Deaths")
        plt.ylabel("Players")
        plt.title("Total Ulduar Deaths P3")
        plt.tight_layout()

    def main(self):
        """
        Main function
        :return: barplot
        """
        fig = plt.figure(figsize=(10, 20))
        plt.xticks(rotation=90)
        plt.tight_layout()
        anim = FuncAnimation(fig, self.animate, interval=1000, frames=200)
        plt.show()

    def save_plot(self):
        """
        Saves the plot as a gif
        :return: gif
        """
        fig = plt.figure(figsize=(10, 20))
        plt.xticks(rotation=90)
        plt.tight_layout()
        anim = FuncAnimation(fig, self.animate, interval=1000, frames=200)
        anim.save(r'C:\Users\torbj\ShameOnDanzi.gif', writer='imagemagick', fps=1)

if __name__ == "__main__":
    animater = BarplotAnimater()
    animater.main()
    # animater.save_plot()