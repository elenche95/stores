import numpy as np
import pandas as pd
from pathlib import Path
import pickle



def metric_adam(preds, actuals):
    '''
    Function defined to get metric for test data
    '''
    print(preds.shape)
    print(actuals.shape)
    
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    
    print(preds.shape)
    print(actuals.shape)
    assert preds.shape == actuals.shape
    
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])



# load the model and run prediction
def predict_with_metric(X_test, str_y_test):
    '''
    Function to get final prediction and metric from test data
    parameters:
    
    - X_test: is the test set dataframe # need to make sure the steps before return a dataframe
    - str_y_test: is the string for the file path to the actual results
    
    Returns:
     - the test metric
     - predictions
    
    '''
    path_y_test = Path(str_y_test)
    y_test = pd.read_csv(path_y_test, index_col=0)
    filename = 'model.pkl'
    model = pickle.load(open(filename, 'rb'))
    pred = model.predict(X_test)
    print(len(pred))

    metric_test = metric_adam(pred, y_test.values)
    
    return metric_test, pred
