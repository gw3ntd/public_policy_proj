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

pd.set_option('display.max_columns', None)

scaler = StandardScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

pca = PCA()
pca.fit(df_scaled)
pca_results = vs.pca_results(df_scaled, pca)
# print(pca_results)
# plt.show()

pca = PCA(n_components=2)
pca.fit(df_scaled)

reduced_data = pca.transform(df_scaled)
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

vs.biplot(df_scaled, reduced_data, pca)
# plt.show()

pc_df = pd.concat([reduced_data, y], axis=1)
print(pc_df.corr()['g_va death rate'])

'''
Output:
Dimension 1       -0.764236
Dimension 2       -0.233349
g_va death rate    1.000000

Analysis:
PC1 has large positive and large negative.
It kinda represents socioeconomic status. 
Like, Bach degree, fam income, house income, 
vma income are all posiive while all the
poverties are negative. 
Because it has a high correlation, it means
that states w/ a lower socioeconomic status
have a higher gun violence death rate. 
'''
