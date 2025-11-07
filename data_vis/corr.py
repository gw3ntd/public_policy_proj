import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# df_target = pd.read_csv('eda/percents.csv')

# y = df_target[['g_va death rate']]

x1 = pd.read_csv('eda/x1.csv')

# print(x1.head())

matrix = x1.corr()

print(matrix)


plt.figure(figsize=(10,7))
sns.heatmap(matrix, annot=True, cmap="Spectral", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()