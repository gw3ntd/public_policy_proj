import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

x1 = pd.read_csv('eda/x1.csv')

x2 = pd.read_csv('eda/x2.csv')

x3 = pd.read_csv('eda/x3.csv')


def make_heatmap(x, cmap='Spectral'):
    matrix = x.corr()
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

make_heatmap(x1)
make_heatmap(x2)
make_heatmap(x3)