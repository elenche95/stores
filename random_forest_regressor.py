import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeRegressor

# importing the data frame
df = pd.read_csv('./transformed.csv')

df.reindex(np.random.permutation(df.index))

df = df[(df.Sales != 0) & (df.Sales.isna() != True)]

n_df = df.shape[0]
n_dev = n_df/10

training_data = df.loc[:n_df-n_dev, :]
dev_data = df.loc[n_df-n_dev:, :]

sorted_id_training = training_data.loc[:, 'sorted id']
sales_training = training_data.loc[:, 'Sales']
sales_training = sales_training.to_numpy()

sorted_id_training = sorted_id_training.to_numpy()
sorted_id_training = sorted_id_training.reshape(-1, 1)


regr_1 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(sorted_id_training, sales_training)

sorted_id_dev = dev_data.loc[:, 'sorted id']
sorted_id_dev = sorted_id_dev.to_numpy()
sorted_id_dev = sorted_id_dev.reshape(-1, 1)
sales_dev = dev_data.loc[:, 'Sales']
sales_dev = sales_dev.to_numpy()
prediction = regr_1.predict(sorted_id_dev)
error = (sales_dev-prediction)**2/sales_dev**2
np.sqrt(np.mean(error))*100
