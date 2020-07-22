from clean2 import clean
import pandas as pd
import numpy as np
df = clean('./data/test.csv')

cat_variables = ['StateHoliday_new', 'StoreType', 'Assortment', 'PromoInterval']

df_final = pd.get_dummies(df, columns = cat_variables, drop_first=True)

X_test = df_final.drop(labels='Sales', axis=1)
Y_test = df_final.Sales


from joblib import dump, load

dtree = load('./dtree_final.joblib')

predictions = dtree.predict(X_test)

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


print(f'RSMPE: {round(metric(np.array(predictions), np.array(Y_test)), 2)}')   
