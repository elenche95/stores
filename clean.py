import pandas as pd
from pathlib import Path

def clean(str_path):
#This function takes the string argument for the path of the train.csv input str_path
    path = Path(str_path)
    train = pd.read_csv(path, parse_dates=True, index_col='Date')
    train['year'] = train.index.year
    train['month'] = train.index.year
    train['week'] = train.index.year
    train['day'] = train.index.year
    train['day_week'] = train.index.dayofweek
    train['month_start'] = train.index.is_month_start.astype(int)
    train['month_end'] = train.index.is_month_end.astype(int)
    holidays = train['StateHoliday'].replace({0 : 0, None : 0}).astype(str)
    train['StateHoliday_new'] = holidays
    
    
    store = pd.read_csv('./data/store.csv')
    store['Promo2SinceWeek'] = store.Promo2SinceWeek.fillna(0)
    store['Promo2SinceYear'] = store.Promo2SinceYear.fillna(0)
    store['PromoInterval'] = store.PromoInterval.fillna('None')
    store['CompetitionDistance'] = store.CompetitionDistance.fillna(round(store.CompetitionDistance.mean()))
    store['CompetitionOpenSinceMonth'] = store.CompetitionOpenSinceMonth.fillna(round(store.CompetitionOpenSinceMonth.mean()))
    store['CompetitionOpenSinceYear'] = store.CompetitionOpenSinceYear.fillna(round(store.CompetitionOpenSinceYear.mean()))
    
    merged = train.merge(store, on='Store', how='inner')
    
    return merged 
    
