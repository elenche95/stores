import numpy as np
import pandas as pd
from clean import clean


def creating_conversion_table():
    '''creating a table for sorting store ids based on the sale'''
    
    df_original = clean('./data/train.csv')
    store_id_vs_average_sale = df_original.groupby('Store').agg('mean')
    sorted = store_id_vs_average_sale.sort_values(by='Sales')
    sorted['new_id'] = np.array(range(1,sorted.shape[0]+1))
    return pd.DataFrame(sorted.new_id)


def creating_df_with_sorted_id(df, conversion):
    ''' 
    creates a new column for with a new id for each store

    Key argument -- a conversion table which gets the original id and gives back the new id
    '''
    df_v_1 = df
    df_v_1['sorted id'] = [conversion.loc[df.Store[id]] for id in range(df.shape[0])]

    return df_v_1
