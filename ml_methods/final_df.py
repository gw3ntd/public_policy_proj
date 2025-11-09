import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

'''
I am first going to make the final dataframe with the features that I want.
After doing EDA and visualization, it seems as if only some features are relevant 
to the gun violence discussion. The final dataset has: Fams Below Pov %, 
Ppl (<150% Of Pov) %, Ppl (Below Poverty) %, % At Least Bachelor's Degree, 
Median Fam Income (Scaled), Median Household Income (Scaled), Vast Majority Income (Scaled), 
and Language Isolation %. All of these features had higher correlations with
the gun violence death rate. 
'''

pd.set_option('display.max_columns', None)
final_df = pd.read_csv('eda/percents.csv')

# dropping district of columbia
final_df.drop([8], axis=0, inplace=True)

# dropping the features that I don't want
final_df.drop(['State', 'Below 9th Ed %', 'Below HS Ed %', 'Unemployed %', 'avg GDP from 2019-2023',
          'avg realGDP from 2019-2023'], axis=1, inplace=True)


# changing the object types to floats
var = ['Median Fam Income (Dollars)', 'Median Household Income (Dollars)', 
       'Vast Majority Income (Dollars)']

for col in var:
    final_df[col] = (
        final_df[col]
        .astype(str)
        .str.replace('[\$,]', '', regex=True)
        .astype(float)
    )

scaler = StandardScaler()

# scaling some features
final_df[['Median Fam Income (Dollars)']] = scaler.fit_transform(final_df[['Median Fam Income (Dollars)']])
final_df[['Median Household Income (Dollars)']] = scaler.fit_transform(final_df[['Median Household Income (Dollars)']])
final_df[['Vast Majority Income (Dollars)']] = scaler.fit_transform(final_df[['Vast Majority Income (Dollars)']])

# renaming the scaled features
final_df.rename(columns={'Median Fam Income (Dollars)': 'Median Fam Income (Scaled)'}, inplace=True)
final_df.rename(columns={'Median Household Income (Dollars)': 'Median Household Income (Scaled)'}, inplace=True)
final_df.rename(columns={'Vast Majority Income (Dollars)': 'Vast Majority Income (Scaled)'}, inplace=True)

print(final_df.head())

# the final df
final_df.to_csv('final_df.csv', index=False)