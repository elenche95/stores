from clean2 import clean
import pandas as pd
import numpy as np

print('Cleaning and loading test set ...')
df = clean('./data/test.csv')

print('Model loaded, applying transformations')
cat_variables = ['StateHoliday_new', 'StoreType', 'Assortment', 'PromoInterval']

print(f'Number of rows in test.csv: {len(df)}')

df_final = pd.get_dummies(df, columns = cat_variables, drop_first=True)

X_test = df_final.drop(labels='Sales', axis=1)
Y_test = df_final.Sales


from joblib import dump, load

print('Loading 8GB of Random Forest Regressor model...')
dtree = load('./dtree_final.joblib')

print('Predicting Sales for the test set')
predictions = dtree.predict(X_test)

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


print(f'Evaluation completed for {len(df)} rows')
print(f'RSMPE: {round(metric(np.array(predictions), np.array(Y_test)), 2)}%')   
