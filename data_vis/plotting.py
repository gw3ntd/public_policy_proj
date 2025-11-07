import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


'''
In this file, I am visualizing all the features in my percents dataset
to get a better understanding of how each feature relates to the 
target
'''

df = pd.read_csv('eda/percents.csv')

# all of the columns listed out for ease of reference

var = ['Fams Below Pov %', 'Ppl (<150% Of Pov) %', 'Ppl (Below Poverty) %', 
       'Below 9th Ed %', 'Below HS Ed %', 'Below Bach Ed %', 
       'Median Fam Income (Dollars)', 'Median Household Income (Dollars)', 
       'Vast Majority Income (Dollars)', 'Language Isolation %', 'Unemployed %', 
       'avg GDP from 2019-2023', 'avg realGDP from 2019-2023']


# changing object types to string types
for col in var:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace('[\$,]', '', regex=True)
        .astype(float)
    )



# first dataframe that only has the %s
X_1 = df.drop(['g_va death rate', 'State', 'Median Fam Income (Dollars)', 'Median Household Income (Dollars)',
               'Vast Majority Income (Dollars)', 'avg GDP from 2019-2023', 'avg realGDP from 2019-2023'], axis=1)

# second dataframe that only has the $
X_2 = df.drop(['Fams Below Pov %', 'Ppl (<150% Of Pov) %', 'Ppl (Below Poverty) %', 
                'Below 9th Ed %', 'Below HS Ed %', 'Below Bach Ed %', 'Language Isolation %', 'Unemployed %',
                'g_va death rate', 'State', 'avg GDP from 2019-2023', 'avg realGDP from 2019-2023'], axis=1)


# third dataframe that deals with gdp
X_3 = df[['avg GDP from 2019-2023', 'avg realGDP from 2019-2023']]

# target, aka gun violence death rate
y = df[['g_va death rate']]


def plot_all(X, y, c='indigo'):
    '''
    This function visualizes all the features
    versus the target given a dataframe X
    '''
    for feature in X:
        plt.figure(figsize=(8, 6))
        plt.scatter(x=X[feature], y=y, c=c)
        plt.xlabel(str(feature))
        plt.ylabel('gun violence death rate')
        plt.title(str(feature) + ' vs. Gun Violence Death Rate')
        plt.show()



scaler = StandardScaler()


# after visualization, it seems as if X_2 needs to be scaled
X_scaled = scaler.fit_transform(X_2)

X_scaled = pd.DataFrame(X_scaled, columns=['Median Fam Income (Dollars)', 'Median Household Income (Dollars)',
                                           'Vast Majority Income (Dollars)'])


# a ln transformation did not do much
X_3['ln_GDP'] = np.log(X_3['avg GDP from 2019-2023'])

X_3['ln_realGDP'] = np.log(X_3['avg realGDP from 2019-2023'])


plot_all(X_1, y, c='chocolate')
plot_all(X_scaled, y, c='lightseagreen')
plot_all(X_3, y)

