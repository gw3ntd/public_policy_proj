import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
from IPython.display import display
from sklearn.decomposition import PCA
import visuals as vs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


'''
PCA
'''

df = pd.read_csv('eda/final_df.csv')
X = df.drop(['g_va death rate'], axis=1)
y = df['g_va death rate']

scaler = StandardScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

pca = PCA()
pca.fit(df_scaled)
pca_results = vs.pca_results(df_scaled, pca)
plt.show()

pca = PCA(n_components=2)
pca.fit(df_scaled)

reduced_data = pca.transform(df_scaled)
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

vs.biplot(df_scaled, reduced_data, pca)
plt.show()