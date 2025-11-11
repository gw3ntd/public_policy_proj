from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('eda/final_df.csv')


pd.set_option('display.max_columns', None)

# print(df.head())


def get_importance(df, model, color='purple'):

    X = df.drop(['g_va death rate'], axis=1)
    y = df['g_va death rate']


    column_names_list = df.columns.tolist()
    column_names_list.pop()

    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Mean r^2 score accross 5 folds for {model}: {np.mean(scores):.3f}")
    print(f"All fold scores for {model}: {scores}\n")

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

get_importance(df, DecisionTreeRegressor(random_state=42))
get_importance(df, RandomForestRegressor(random_state=42), color='teal')
get_importance(df, GradientBoostingRegressor(random_state=42), color='pink')

'''
Mean r^2 score accross 5 folds for DecisionTreeRegressor(random_state=42): 0.236
All fold scores for DecisionTreeRegressor(random_state=42): [ 0.23092503  0.77619647  0.7652932   0.29895134 -0.89273118]

Fams Below Pov %, Score: 0.07457
Ppl (<150% Of Pov) %, Score: 0.00374
Ppl (Below Poverty) %, Score: 0.04395
% At Least Bachelor's Degree, Score: 0.47181
Median Fam Income (Scaled), Score: 0.02827
Median Household Income (Scaled), Score: 0.16602
Vast Majority Income (Scaled), Score: 0.00103
Language Isolation %, Score: 0.21060


Mean r^2 score accross 5 folds for RandomForestRegressor(random_state=42): 0.491
All fold scores for RandomForestRegressor(random_state=42): [ 0.4201826   0.64573927  0.7237883   0.69102035 -0.02674616]

Fams Below Pov %, Score: 0.05428
Ppl (<150% Of Pov) %, Score: 0.03473
Ppl (Below Poverty) %, Score: 0.03383
% At Least Bachelor's Degree, Score: 0.30526
Median Fam Income (Scaled), Score: 0.18209
Median Household Income (Scaled), Score: 0.11751
Vast Majority Income (Scaled), Score: 0.11447
Language Isolation %, Score: 0.15783


Mean r^2 score accross 5 folds for GradientBoostingRegressor(random_state=42): 0.432
All fold scores for GradientBoostingRegressor(random_state=42): [ 0.2980692   0.6816932   0.73679643  0.70031228 -0.25924952]

Fams Below Pov %, Score: 0.03618
Ppl (<150% Of Pov) %, Score: 0.03724
Ppl (Below Poverty) %, Score: 0.02374
% At Least Bachelor's Degree, Score: 0.39926
Median Fam Income (Scaled), Score: 0.10813
Median Household Income (Scaled), Score: 0.13981
Vast Majority Income (Scaled), Score: 0.06105
Language Isolation %, Score: 0.19459
'''