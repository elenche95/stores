import numpy as np
import pandas as pd
#from pathlib import Path
import pickle

def metric_adam(preds, actuals):
    '''
    Function defined to get metric for test data
    '''
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)

    assert preds.shape == actuals.shape
    
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])

# load the model and run prediction
def predict_with_metric(X_test, y_test):
    '''
    Function to get final prediction and metric from test data
    
    Parameters:
    - X_test: is the test set dataframe # need to make sure the steps before return a dataframe
    - y_test: is the dataframe of the actual results
    
    Returns:
     - the test metric
     - predictions
    
    '''
    filename = 'model.pkl'
    model = pickle.load(open(filename, 'rb'))
    pred = model.predict(X_test)

    metric_test = metric_adam(pred, y_test.values)
    print("The error is {}".format(metric_test))
    
    return metric_test, pred
