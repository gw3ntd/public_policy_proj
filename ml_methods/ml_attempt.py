from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda/final_df.csv')


pd.set_option('display.max_columns', None)

# print(df.head())

def get_importance(df, model, color='purple'):
    X = df.drop(['g_va death rate'], axis=1)
    y = df[['g_va death rate']]

    column_names_list = df.columns.tolist()
    column_names_list.pop()

    model.fit(X, y)

    importance = model.feature_importances_

    for i, v in enumerate(importance):
        print(f'{column_names_list[i]}, Score: {v:.5f}')
    plt.bar([x for x in range(len(importance))], importance, color=color)
    plt.xticks([x for x in range(len(importance))], column_names_list, rotation=45, ha='right')
    plt.title(str(model) + ' Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Feature Importance Scores')
    plt.tight_layout()
    plt.show()

get_importance(df, DecisionTreeRegressor())
get_importance(df, RandomForestRegressor(), color='teal')

'''
Decision Tree

Fams Below Pov %, Score: 0.03006
Ppl (<150% Of Pov) %, Score: 0.04787
Ppl (Below Poverty) %, Score: 0.00034
% At Least Bachelor's Degree, Score: 0.47104
Median Fam Income (Scaled), Score: 0.02941
Median Household Income (Scaled), Score: 0.20511
Vast Majority Income (Scaled), Score: 0.00161
Language Isolation %, Score: 0.21456

Random Forest

Fams Below Pov %, Score: 0.05227
Ppl (<150% Of Pov) %, Score: 0.02943
Ppl (Below Poverty) %, Score: 0.02996
% At Least Bachelor's Degree, Score: 0.32617
Median Fam Income (Scaled), Score: 0.15243
Median Household Income (Scaled), Score: 0.12030
Vast Majority Income (Scaled), Score: 0.13241
Language Isolation %, Score: 0.15703
'''