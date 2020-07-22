Mini Competition - Rossman Challenge

This is a repo to predict Sales data from the Rossman Challenge

To run:

1. Import the necessary modules and requirements:


2. Run clean(p, test=True), where p is the string of the path to the test set and save the output dataframe to test_df
test_df = clean(p, test=True)

3. Run the feature engineering function with your test_df and test=True as input and save to feat_df
feat_df = feature_engineering(test_df, test=True)

4. Run model(feat_df, path_y_test), where path_y_test is the string of the path of your actual test data. This returns two objects, the metric and the predictions. 
result = model(feat_df, path_y_test)
