import numpy as np
import pandas as pd
from clean import clean


def creating_conversion_table():
    '''creating a table for sorting store ids based on the sale'''
    
    df_original = clean()
    store_id_vs_average_sale = df_original.groupby('Store').agg('mean')
    sorted = store_id_vs_average_sale.sort_values(by='Sales')
    sorted ['new id'] = np.array(range(1, sorted.shape[0]+1))
    conversion = sorted["new id"]
    return conversion


def creating_df_with_sorted_id(conversion):
    ''' 
    creates a new column for with a new id for each store

    Key argument -- a conversion table which gets the original id and gives back the new id
    '''
    df_original = clean()
    df_v_1 = df_original
    df_v_1['sorted id'] = [conversion.loc[df_original.Store[id]] for id in range(df_original.shape[0])]

    return df_v_1
