import numpy as np
import pandas as pd
from feature_engineering import * 
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor

X_train = pd.read_csv('./X_train.csv', index_col=0)
X_valid = pd.read_csv('./X_valid.csv', index_col=0)
y_train = pd.read_csv('./y_train.csv', index_col=0)
y_valid = pd.read_csv('./y_valid.csv', index_col=0)


def metric_adam(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])

#error metric
def metric(actuals, preds):
    actuals = actuals.values
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return -100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])

# doing grid search for hyperparameters and CV 
parameters = {'n_estimators':[1, 3, 10]}
grid = GridSearchCV(RandomForestRegressor(), parameters, scoring=make_scorer(metric, greater_is_better=False), cv=4)
tmp = grid.fit(X_train.iloc[0:4_000,:], y_train.iloc[0:4_000, :])
print(tmp.cv_results_)

# Testing XGB
parameters = {'learning_rate':[0.3]}
grid = GridSearchCV(XGBRegressor(), parameters, scoring=make_scorer(metric, greater_is_better=False), cv=4)
tmp = grid.fit(X_train.iloc[0:200_000,:], y_train.iloc[0:200_000, :])
print(tmp.cv_results_)

model = XGBRegressor()
model.fit(X_train, y_train)

#save model to a pkl file 

filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))


'''
>>> from custom_scorer_module import custom_scoring_function 
>>> cross_val_score(model,
...  X_train,
...  y_train,
...  scoring=make_scorer(custom_scoring_function, greater_is_better=False),
...  cv=5,
...  n_jobs=-1) 


forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, y_train)
forest_model.feature_importances_

feat_imp = forest_model.feature_importances_
import matplotlib.pyplot as plt
import seaborn as sns
x = X_train.columns
y = feat_imp
temp = np.array([x,y]).T
temp_df = pd.DataFrame(np.log(1/y), index=x)
temp_df.sort_values(by=0, inplace=True)
temp_df.head()
#temp = np.sort(temp)
print(np.shape(temp))
print(temp_df)
fig, ax = plt.subplots(figsize=(6, 15))
#ax.barh(x, np.log(1/y))

sns.barplot(y=temp_df.index, x=0, data=temp_df)
            
            #np.log(1/y), x)


