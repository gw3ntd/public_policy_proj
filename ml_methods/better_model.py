import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


'''
# Fail!
'''


df = pd.read_csv('eda/final_df.csv')

df['Poverty'] = df[['Fams Below Pov %', 'Ppl (<150% Of Pov) %', 'Ppl (Below Poverty) %']].mean(axis=1)
df['Income'] = df[['Median Fam Income (Scaled)', 'Median Household Income (Scaled)', 'Vast Majority Income (Scaled)']].mean(axis=1)

df.drop(['Fams Below Pov %', 'Ppl (<150% Of Pov) %', 'Ppl (Below Poverty) %', 
         'Median Fam Income (Scaled)', 'Median Household Income (Scaled)', 'Vast Majority Income (Scaled)'], 
         axis=1, inplace=True)

X = df.drop(['g_va death rate'], axis=1)
y = df['g_va death rate']

column_names_list = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42)

param_dist = {
    'n_estimators': np.arange(200, 1500),
    'max_depth': [None] + list(np.arange(3, 40)),
    'min_samples_split': np.arange(2, 20),
    'min_samples_leaf': np.arange(1, 10),
    'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
    'bootstrap': [True, False]
}

# search = RandomizedSearchCV(
#     RandomForestRegressor(),
#     param_dist,
#     n_iter=100,
#     cv=5,
#     scoring='r2',
#     n_jobs=-1,
#     random_state=42
# )

# search.fit(X_train, y_train)
# print(search.best_estimator_)


rf = RandomForestRegressor(bootstrap=False, max_depth=10, max_features=0.7,
                      min_samples_leaf=10,
                      min_samples_split=20,
                      n_estimators=np.int64(1136))
rf.fit(X_train, y_train)

scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
print(f"Mean r^2 score accross 5 folds for {rf}: {np.mean(scores):.3f}")
print(f"All fold scores for {rf}: {scores}\n")

rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))

importance = rf.feature_importances_

for i, v in enumerate(importance):
    print(f'{column_names_list[i]}, Score: {v:.5f}')
