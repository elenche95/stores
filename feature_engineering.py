import numpy as np
import pandas as pd
from clean import clean
from feature_engineering_utils import creating_conversion_table
from feature_engineering_utils import creating_df_with_sorted_id
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# 1 creating a new column with store id's which are realted to Sales

# 1.1
print("creating conversion table")
conversion = creating_conversion_table()
# 1.2
df = creating_df_with_sorted_id(conversion)
# 1.3
# Convert the holidays to 1 or 0, but holidays do have a slight different sales numbers
# can One Hot encode later - or label encoding based on order. 
print("converting the holidays")
df['holiday_bool'] = df['StateHoliday_new'].replace({'a': 1, 'b': 1, 'c':1, '0':0 })
df.drop(['StateHoliday_new'], inplace=True, axis=1)

# Set Promo2 Since Week, Promo2 since year to zero since they are only na when Promo2 there is no Promo2 
df['Promo2SinceWeek'].fillna(0, inplace=True)
df['Promo2SinceYear'].fillna(0, inplace=True)



# Splitting train, test, X and y 
X = df.loc[:, df.columns!= 'Sales']
y = df.loc[:, 'Sales']
# K fold to reshuffle the data ?? 
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)


#fill na with average competition distance, month and year, -need to in 
#my_imputer = SimpleImputer(strategy='mean')
#final_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
#final_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

#final_X_train.columns = X_train.columns
#final_X_valid.columns = X_valid.columns

# One Hot Encoding 'StoreType', 'Assortment','PromoInterval'

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

cat_col = [col for col in df.columns if df[col].dtype=='O']

OH_X_train_cols = pd.DataFrame(OH_encoder.fit_transform(X_train[cat_col]))
OH_X_valid_cols = pd.DataFrame(OH_encoder.transform(X_valid[cat_col]))

OH_X_train_cols.index = X_train.index 
OH_X_valid_cols.index = X_valid.index

num_X_train = X_train.drop(cat_col, axis=1)
num_X_valid = X_valid.drop(cat_col, axis=1) 

X_train = pd.concat([num_X_train, OH_X_train_cols], axis=1)
X_valid = pd.concat([num_X_valid, OH_X_valid_cols], axis=1)

# last step save it as csv
df.to_csv('./transformed.csv')
X_train.to_csv('./X_train.csv')
X_valid.to_csv('./X_valid.csv')
y_train.to_csv('./y_train.csv')
y_valid.to_csv('./y_valid.csv')