import numpy as np
import pandas as pd
import pickle
from clean import clean
from feature_engineering_utils import creating_conversion_table
from feature_engineering_utils import creating_df_with_sorted_id
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# 1 creating a new column with store id's which are realted to Sales
def feature_engineering(df, test=False):
    '''
    
    '''
    # creating new index based on Sales
    if test==False:
        print("creating conversion table")
        conversion = creating_conversion_table()
        conversion.to_csv('./conversion.csv')
    else:
        conversion = pd.read_csv('./conversion.csv', index_col=0)
        
    df = creating_df_with_sorted_id(conversion)
    
    # If training we reshuffle and split the data into x and y
    if test==False:
        print('reshuffle the data')
        df.reindex(np.random.permutation(df.index))    
        df.to_csv('./transformed.csv')
    
    print('Splitting X and y')
    X = df.loc[:, df.columns!= 'Sales']
    y = df.loc[:, 'Sales']
    
    if test==False:
        
    # One Hot Encoding 'StateHoliday_new' 'StoreType', 'Assortment','PromoInterval'
        cat_col = [col for col in df.columns if df[col].dtype=='O']
        #cat_col.remove("sorted id") # sorted id is not in this set? why???
        
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        OH_X_cols = pd.DataFrame(OH_encoder.fit_transform(X[cat_col]))
        
    
        OH_X_cols.index = X.index 
        num_X_train = X.drop(cat_col, axis=1)
        X_feat = pd.concat([num_X_train, OH_X_cols], axis=1)
        
        # saving the encoder
        with open("OH_encoder.pkl", "wb") as f: 
            pickle.dump(OH_encoder, f)
    
        #Save it as csv
        X_feat.to_csv('./X_feat.csv')
        print(X_feat.head())
        y.to_csv('./y.csv')
        
        return X_feat, y
        
    else:
        X_test = one_hot_test(X)
        print(X_test.head())
        X_test.to_csv('./X_test.csv')
        y.to_csv('./y_test.csv')
        
        return X_test, y

def one_hot_test(df):    
    # define the categorical variable columns
    cat_col = [col for col in df.columns if df[col].dtype=='O']
    cat_col.remove("sorted id")
    
    # load the transformer
    with open('OH_encoder.pkl', 'rb') as f:
        OH_encoder = pickle.load(f)

    # make new df
    print(df)
    #import pdb; pdb.set_trace()
    #if "sorted id" in df.columns:
    OH_df_cols = pd.DataFrame(OH_encoder.transform(df[cat_col]))
    OH_df_cols.index = df.index
    num_df = df.drop(cat_col, axis=1)
    df_with_OH = pd.concat([num_df, OH_df_cols], axis=1) 
    
    return df_with_OH

