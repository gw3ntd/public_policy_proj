import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


'''
Here, I am finding the feature importance scores
using GradientBoostingRegressor. 
'''


df = pd.read_csv('eda/final_df.csv')


X = df.drop(['g_va death rate'], axis=1)
y = df['g_va death rate']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)


gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)

importance = gb.feature_importances_
column_names_list = X.columns.tolist()

# Printing the scores
for i, v in enumerate(importance):
    print(f'{column_names_list[i]}, Score: {v:.5f}')

# Putting the data in a dataframe and converting it to a csv file for later use
feature_importance_df = pd.DataFrame({'Feature': column_names_list, 'GB': importance})    
print("Feature importance:\n", feature_importance_df)

feature_importance_df.to_csv('ml_methods/GB.csv', index=False)

'''
Fams Below Pov %, Score: 0.03257
Ppl (<150% Of Pov) %, Score: 0.02407
Ppl (Below Poverty) %, Score: 0.02293
% At Least Bachelor's Degree, Score: 0.56215
Median Fam Income (Scaled), Score: 0.02281
Median Household Income (Scaled), Score: 0.05274
Vast Majority Income (Scaled), Score: 0.08948
Language Isolation %, Score: 0.19326
'''