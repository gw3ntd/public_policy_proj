import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance



df = pd.read_csv('eda/final_df.csv')


X = df.drop(['g_va death rate'], axis=1)
y = df['g_va death rate']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

ridge = Ridge(alpha=1e-2).fit(X_train, y_train)
print(ridge.score(X_test, y_test))

r = permutation_importance(ridge, X_test, y_test,
                           n_repeats=30,
                           random_state=0)

column_names_list = df.columns.tolist()

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{column_names_list[i]:<8}" + ': '
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")

''' 
Ppl (Below Poverty) %: 31.006 +/- 6.217
Ppl (<150% Of Pov) %: 22.432 +/- 4.443
Fams Below Pov %: 8.531 +/- 1.567
Vast Majority Income (Scaled): 0.874 +/- 0.381
Language Isolation %: 0.699 +/- 0.287
Median Fam Income (Scaled): 0.465 +/- 0.169
'''        
