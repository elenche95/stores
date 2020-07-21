import numpy as np
import pandas as pd
import sklearn
from random_forest_regressor_functions import random_forest_regressor
from random_forest_regressor_functions import base_line
features = ['sorted id', 'Promo', 'Customers']
results = {'mean': base_line('./transformed.csv')}
for i in range(0, len(features)):
    input_features = features[:i+1]
    error = random_forest_regressor('./transformed.csv', 5, input_features)
    results[features[i]] = error


import matplotlib.pyplot as plt


plt.bar(range(len(results)), list(results.values()), align='center')
plt.xticks(range(len(results)), list(results.keys()))

plt.show()
