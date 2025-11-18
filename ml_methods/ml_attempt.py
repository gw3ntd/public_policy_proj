from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

'''
In this file, I am attempting to find what features
find gun violence death rate the best as well as 
finding feature importance
'''

df = pd.read_csv('eda/final_df.csv')

pd.set_option('display.max_columns', None)

def get_importance(df, model, color='purple'):
    '''
    This function finds the importance score
    of each feature given a certain model. It
    also can generate a graph. 
    '''

    X = df.drop(['g_va death rate'], axis=1)
    y = df['g_va death rate']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42)


    column_names_list = X.columns.tolist()

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"Mean r^2 score accross 5 folds for {model}: {np.mean(scores):.3f}")
    print(f"All fold scores for {model}: {scores}\n")

    model.fit(X_train, y_train)

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

    feature_importance_df = pd.DataFrame({'Feature': column_names_list, 'RF': importance})    
    return feature_importance_df

# get_importance(df, DecisionTreeRegressor(random_state=42))
f = get_importance(df, RandomForestRegressor(random_state=42), color='teal')
f.to_csv('ml_methods/RF.csv', index=False)
# print("Feature importance:\n", f)
# get_importance(df, GradientBoostingRegressor(random_state=42), color='pink')


new = df.drop(['g_va death rate'], axis=1)
columns = new.columns.tolist()

# creating a list of every possible combination of four variables
combos = []
for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        for k in range(j + 1, len(columns)):
            for l in range(k + 1, len(columns)):
                combos.append([columns[i], columns[j], columns[k], columns[l]])


# creating a list of every possible combination of three variables
other_combo = []
for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        for k in range(j + 1, len(columns)):
            other_combo.append([columns[i], columns[j], columns[k]])


def all_iterations(df, combos):
    '''
    This function takes in all of these possible combinations
    and finds the 5-fold cross validation score of a 
    random forest model for each. It adds each combonation
    of variables with a cv score above 0.5 to a list.
    '''
    idx_list = []
    new = df.drop(['g_va death rate'], axis=1)
    for i in range(len(combos)):
        X = new[combos[i]]
        y = df['g_va death rate']
        rf = RandomForestRegressor()
        scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
        if np.mean(scores) >= 0.5:
            idx_list.append(i)
            # print(f"{i}: {combos[i]}")
            # print(f"Mean r^2 score accross 5 folds for {rf}: {np.mean(scores):.3f}")
            # print(f"All fold scores for {rf}: {scores}\n")
    return idx_list

# L1 = all_iterations(df, combos)
# L2 = all_iterations(df, other_combo)

# print(L1)
# print(L2)

def div_importance(df, combos, idx):
    '''
    This function finds the feature importance
    for all the combos that had a 5-fold 
    cv score over 0.5 and prints them
    out alongside their corresponding
    index in the combos list
    '''
    new = df.drop(['g_va death rate'], axis=1)
    rf = RandomForestRegressor()
    for num in idx:
        X = new[combos[num]]
        y = df['g_va death rate']
        rf.fit(X, y)
        importance = rf.feature_importances_
        print('\n' + str(num))
        for i, v in enumerate(importance):
            print(f'{combos[num][i]}, Score: {v:.5f}')

# div_importance(df, combos, L1)
# div_importance(df, other_combo, L2)

