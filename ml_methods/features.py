import pandas as pd
import matplotlib.pyplot as plt

'''
I have the feature importance scores for GradientBoosting, RandomForest, 
Mutual Information, and Permutation Importance and have stored them in csv 
files. I then combined all of the csv files into a pandas dataframe. 

After I graphed these, mutual information seemed unlike the values
of the other methods. So, for now, I chose to leave it off the
graph. 
'''


gb = pd.read_csv('ml_methods/GB.csv')
mi = pd.read_csv('ml_methods/MI.csv')
rf = pd.read_csv('ml_methods/RF.csv')
perm = pd.read_csv('ml_methods/PERM.csv')

features = pd.merge(gb, rf, on='Feature', how='outer') 
features = pd.merge(features, perm, on='Feature', how='outer')
# features = pd.merge(features, perm, on='Feature', how='outer')

# Plotting the features and their importances
features = features.set_index('Feature')
features.plot(kind='bar', figsize=(8, 5)) 
plt.title('Feature Importance Scores Across all Methods')
plt.ylabel('Feature Importance')
plt.xlabel('Features')
plt.xticks(rotation=45, ha='right') 
plt.legend(title='Value Type')
plt.tight_layout()
plt.show()
