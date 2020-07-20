import numpy as np
import pandas as pd
from clean import clean
from feature_engineering_utils import creating_conversion_table
from feature_engineering_utils import creating_df_with_sorted_id

# 1 creating a new column with store id's which are realted to Sales

# 1.1
conversion = creating_conversion_table()
# 1.2
df_v_1 = creating_df_with_sorted_id(conversion)
# 1.3


# last step save it as csv

df_v_1.to_csv('./transformed.csv')