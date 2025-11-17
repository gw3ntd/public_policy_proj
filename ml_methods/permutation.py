import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor



df = pd.read_csv('eda/final_df.csv')


X = df.drop(['g_va death rate'], axis=1)
y = df['g_va death rate']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)
print(rf.score(X_test, y_test))

r = permutation_importance(rf, X_test, y_test,
                           n_repeats=30,
                           random_state=0, scoring='r2')

column_names_list = X.columns.tolist()

for i in r.importances_mean.argsort()[::-1]:
    print(f"{column_names_list[i]:<8}" + ': '
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}")
    
feature_importance_df = pd.DataFrame({'Feature': column_names_list, 'Permutation': r.importances_mean})    
print("Feature importance:\n", feature_importance_df)

feature_importance_df.to_csv('ml_methods/PERM.csv', index=False)

''' 
% At Least Bachelor's Degree: 0.386 +/- 0.228
Vast Majority Income (Scaled): 0.109 +/- 0.055
Language Isolation %: 0.084 +/- 0.075
Median Fam Income (Scaled): 0.076 +/- 0.038
Fams Below Pov %: 0.042 +/- 0.010
Ppl (<150% Of Pov) %: 0.040 +/- 0.014
Ppl (Below Poverty) %: 0.015 +/- 0.007
Median Household Income (Scaled): 0.014 +/- 0.011
'''        
