import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeRegressor


def random_forest_regressor(inputfile, tree_depth, features):

    df = pd.read_csv(inputfile)

    df.reindex(np.random.permutation(df.index))

    df = df[(df.Sales != 0) & (df.Sales.isna() != True)]
    df = df[(df.Promo.isna() != True)]
    df = df[(df.Customers.isna() != True)]
    df = df[(df.DayOfWeek.isna() != True)]
    n_df = df.shape[0]
    n_dev = n_df/10

    training_data = df.loc[:n_df-n_dev, :]
    dev_data = df.loc[n_df-n_dev:, :]

    y_training = training_data.loc[:, 'Sales']
    y_training = y_training.to_numpy()
    y_training = y_training.reshape(-1, 1)

    x_training = training_data.loc[:, features]
    x_training = x_training.to_numpy()

    regr_1 = DecisionTreeRegressor(max_depth=tree_depth)
    regr_1.fit(x_training, y_training)

    x_dev = dev_data.loc[:, features]
    x_dev = x_dev.to_numpy()
    y_dev = dev_data.loc[:, 'Sales']
    y_dev = y_dev.to_numpy()

    prediction = regr_1.predict(x_dev)
    error = (y_dev-prediction)**2/y_dev**2

    return np.sqrt(np.mean(error))*100


def base_line(inputfile):

    df = pd.read_csv(inputfile)

    df.reindex(np.random.permutation(df.index))

    df = df[(df.Sales != 0) & (df.Sales.isna() != True)]
    df = df[(df.Promo.isna() != True)]
    df = df[(df.Customers.isna() != True)]

    n_df = df.shape[0]
    n_dev = n_df/10

    training_data = df.loc[:n_df-n_dev, :]
    dev_data = df.loc[n_df-n_dev:, :]

    y_training = training_data.loc[:, 'Sales']
    y_training = y_training.to_numpy()
    y_training = y_training.reshape(-1, 1)

    y_dev = dev_data.loc[:, 'Sales']
    y_dev = y_dev.to_numpy()

    prediction = np.mean(y_training)
    error = (y_dev-prediction)**2/y_dev**2

    return np.sqrt(np.mean(error))*100
