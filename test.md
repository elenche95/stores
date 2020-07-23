Mini Competition - Rossman Challenge

This is a repo to predict Sales data from the Rossman Challenge

To run:

1. Import the necessary modules and requirements:
pip install -r requirements.txt
from clean import clean
from feature_engineering import feature_engineering
from model import predict_with_metric

2. Run clean(p), where p is the string of the path to the test set and save the output dataframe to test_df
df_test = clean(p)

3. Run the feature engineering function with your test_df and test=True as input and save to feat_df
X_test, y_test = feature_engineering(df_test, test=True)

4. Run model(feat_df, path_y_test), where path_y_test is the string of the path of your actual test data. This returns two objects, the metric and the predictions. 
metric, pred = predict_with_metric(X_test, y_test)
