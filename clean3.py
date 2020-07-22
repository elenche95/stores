import pandas as pd
from pathlib import Path


def store_clean():
    store = pd.read_csv("./data/store.csv")
    store['Promo2SinceWeek'] = store.Promo2SinceWeek.fillna(0)
    store['Promo2SinceYear'] = store.Promo2SinceYear.fillna(0)
    store['PromoInterval'] = store.PromoInterval.fillna('None')
    store['CompetitionDistance'] = store.CompetitionDistance.fillna(round(store.CompetitionDistance.mean()))
    store['CompetitionOpenSinceMonth'] = store.CompetitionOpenSinceMonth.fillna(round(store.CompetitionOpenSinceMonth.mean()))
    store['CompetitionOpenSinceYear'] = store.CompetitionOpenSinceYear.fillna(round(store.CompetitionOpenSinceYear.mean()))
    return store


# Lambda functions for data cleaning

def customers_nan(sales, customers):
        mean_sales_customers = 9.38648627379399 # (train['Sales'] / train['Customers']).mean()
        if customers != customers:  # Clever way of verifying if an object  is null
            return round(sales / mean_sales_customers)
        else: 
            return customers


# Cleaning training dataset

def train_clean(path, test):
    
    # Setting index and time auxiliary columns
    train = pd.read_csv(path, parse_dates=True, index_col='Date')
    train.index = pd.to_datetime(train.index)
    train['year'] = train.index.year
    train['month'] = train.index.month
    train['week'] = train.index.week
    train['day'] = train.index.day
    train['day_week'] = train.index.dayofweek
    train['month_start'] = train.index.is_month_start
    train['month_end'] = train.index.is_month_end
    holidays = train['StateHoliday'].replace({0 : 0, None : 0}).astype(str)
    train['StateHoliday_new'] = holidays
    
    # Dropping all the rows with sales == 0 or sales == nan
    train = train[(train.Sales != 0) & (train.Sales.isna() != True)]
    
    # Drop missing Store
    
    train = train[(train.Store.isna() != True)]
    
    # Drop DayOfWeek as this information is in day_week
    
    train.drop(axis=1, labels='DayOfWeek', inplace=True)
    
    #Substitute missing values in customers by the mean of Sales and Customers
    #does not work for test data  
    if test==False:
        train['Sales_per_Customer'] = train.apply(lambda x: customers_nan(x['Sales'], x['Customers']), axis=1)
    
    # Set Open Column to 1 OR DROP IT
    
    train['Open'] = train.loc[:, 'Open'].apply(lambda x: 1)
    
    # Fill train['Promo'] with 0
    
    train['Promo'] = train.Promo.fillna(0)
    
    # Check if you can improve the missing values on State Holiday and School Holiday
    
    train.drop(axis=1, labels= ['StateHoliday'], inplace=True)
    
    train['SchoolHoliday'] = train['SchoolHoliday'].fillna(0)

    return train


def clean(str_path, test=False):
    """ This is a function to clean data,
    parameters:
    - str_path is a path of the data
    - test=True if this is a test set
    
    """
    #This function takes the string argument for the path of the train.csv input str_path
    path = Path(str_path)
    train = train_clean(path, test)
    store = store_clean()
    
    merged = train.merge(store, on='Store', how='inner')
    
    
    return merged 
    

def test_clean(test):
    # First step in the test set is to get rid of the values Sales = 0 | Sales = nan
    test = test[(test.Sales != 0) & (test.Sales.isna() != True)]
