from sklearn.feature_selection import mutual_info_regression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib as plt
from sklearn.feature_selection import SelectPercentile


df = pd.read_csv('eda/final_df.csv')

print(df.info())

X = df.drop(['g_va death rate'], axis=1)
y = df['g_va death rate']

mutual_info = mutual_info_regression(X, y)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X.columns
print(mutual_info.sort_values(ascending=False))

'''
Median Fam Income (Scaled)          0.518327
Vast Majority Income (Scaled)       0.405067
Median Household Income (Scaled)    0.373464
Ppl (Below Poverty) %               0.340521
% At Least Bachelor's Degree        0.330106
Fams Below Pov %                    0.294105
Ppl (<150% Of Pov) %                0.273153
Language Isolation %                0.193789
'''

selected_top_columns = SelectPercentile(mutual_info_regression, percentile=20)
selected_top_columns.fit(X.fillna(0), y)
selected_top_columns.get_support()
print(X.columns[selected_top_columns.get_support()])

'''
Index(['Median Fam Income (Scaled)', 
'Vast Majority Income (Scaled)'], dtype='object')

So these are the features in the top 20th percentile
'''