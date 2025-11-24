import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

x1 = pd.read_csv('eda/x1.csv')

x2 = pd.read_csv('eda/x2.csv')

x3 = pd.read_csv('eda/x3.csv')

final = pd.read_csv('eda/final_df.csv')


def make_heatmap(x, cmap='PiYG', title="Correlation Heatmap"):
    matrix = x.corr()
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()

make_heatmap(x1, title="Correlation Heatmap for Socioeconomic Features")
make_heatmap(x2, title="Correlation Heatmap for Income-Related Features")
make_heatmap(x3, title="Correlation Heatmap for GDP")
# make_heatmap(final)