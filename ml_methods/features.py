import pandas as pd
import matplotlib.pyplot as plt


gb = pd.read_csv('ml_methods/GB.csv')
mi = pd.read_csv('ml_methods/MI.csv')
rf = pd.read_csv('ml_methods/RF.csv')
perm = pd.read_csv('ml_methods/PERM.csv')

features = pd.merge(gb, mi, on='Feature', how='outer')
features = pd.merge(features, rf, on='Feature', how='outer')
features = pd.merge(features, perm, on='Feature', how='outer')

print(features)

features = features.set_index('Feature')
features.plot(kind='bar', figsize=(8, 5)) 
plt.title('Feature Importance Scores Across all Methods')
plt.ylabel('Feature Importance')
plt.xlabel('Features')
plt.xticks(rotation=45, ha='right') 
plt.legend(title='Value Type')
plt.tight_layout()
plt.show()
