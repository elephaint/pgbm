"""
   Copyright (c) 2021 Olivier Sprangers 

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   https://github.com/elephaint/pgbm/blob/main/LICENSE

"""
#%% Import packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 15)
#%% Reduce memory
def reduce_mem(df):
    cols = df.columns
    for col in cols:
        col_dtype = df[col].dtype 
        try:               
            if col_dtype == pd.Int8Dtype():
                df[col] = df[col].astype('int8')
            elif col_dtype == pd.Int16Dtype():
                df[col] = df[col].astype('int16')
            elif col_dtype == pd.Int32Dtype():
                df[col] = df[col].astype('int16')    
            elif col_dtype == pd.Int64Dtype():
                df[col] = df[col].astype('int16')
        except:
            pass
        if col_dtype == 'int64':
            df[col] = df[col].astype('int16')
        elif col_dtype == 'float64':
            if 'sales_lag' in col:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('float32')
            
    return df
#%% 1) Read datasets
df_calendar = pd.read_csv('pgbm/datasets/m5/calendar.csv').fillna("None").convert_dtypes() # NaN's only in event fields, so fill with None string
df_sales = pd.read_csv('pgbm/datasets/m5/sales_train_evaluation.csv').convert_dtypes()
df_prices = pd.read_csv('pgbm/datasets/m5/sell_prices.csv').convert_dtypes()
#%% 2) Label encoding of categorical information sales data
#https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
df_itemids = df_sales[['id','item_id','dept_id','cat_id','store_id','state_id']].copy()
for col in df_itemids.columns:
    df_itemids[col+'_enc'] = LabelEncoder().fit_transform(df_itemids[col])
    df_itemids[col+'_enc'] = df_itemids[col+'_enc'].astype('int16')
    
df_itemids = df_itemids[['id_enc', 'item_id_enc', 'dept_id_enc', 'cat_id_enc', 'store_id_enc', 'state_id_enc', 'id',
       'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
#%% 2a) Add label encoded item_ids to prices
df_prices = df_prices.merge(df_itemids[['store_id','item_id','item_id_enc','store_id_enc','dept_id_enc','cat_id_enc']], how='left', left_on=['store_id','item_id'], right_on=['store_id','item_id'])
# Assert that we don't create NaNs - in other words, every item in df_prices is represented in itemids
assert df_prices.isnull().sum().sum() == 0
df_prices = df_prices[['wm_yr_wk', 'sell_price', 'item_id_enc', 'store_id_enc', 'dept_id_enc']].copy()
# Add price changes
group = df_prices.groupby(['item_id_enc','store_id_enc'])['sell_price']
df_prices['sell_price_change'] = group.shift(0) / group.shift(1) - 1
df_prices['sell_price_change'] = df_prices['sell_price_change'].fillna(0)
# Add normalized selling prices 
def add_normprices(df, col, group):
    df[col] = (group.shift(0) - group.transform(np.mean)) / group.transform(np.std)
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(0)
    
    return df

df_prices = add_normprices(df_prices, 'sell_price_norm_item', df_prices.groupby(['item_id_enc'])['sell_price']) # Normalized price per item
df_prices = add_normprices(df_prices, 'sell_price_norm_dept', df_prices.groupby(['dept_id_enc'])['sell_price']) # Normalized price per department
df_prices = df_prices.drop(columns = ['dept_id_enc'])
# Add weeks on sale
df_prices['weeks_on_sale'] = df_prices.groupby(['item_id_enc','store_id_enc']).cumcount() + 1
# Reduce mem
df_prices = reduce_mem(df_prices)
#%% 3) Label encoding of categorical information in calendar data
df_events = df_calendar[['event_name_1','event_type_1','event_name_2','event_type_2']].copy()
cols = df_events.columns
for col in cols:
    df_events[col+'_enc'] = LabelEncoder().fit_transform(df_events[col])
    df_events[col+'_enc'] = df_events[col+'_enc'].astype('int16')
df_events = df_events.drop(columns = cols)

df_events = pd.concat( (df_calendar[['date','d','wm_yr_wk','snap_CA', 'snap_TX','snap_WI']], df_events), axis = 1)
cols = df_events.columns[3:].tolist()
df_events[cols] = df_events[cols].astype('int8')
df_events['date'] = pd.to_datetime(df_events['date'])
# Reduce mem
df_events = reduce_mem(df_events)
#%% 4) Create main df
# Add zero columns for test period to sales data
new_columns = df_events['d'][-28:].tolist()
df_sales = df_sales.assign(**pd.DataFrame(data = 0, columns=new_columns, index=df_sales.index))
# Add item ids and stack
df = pd.concat((df_itemids['id_enc'], df_sales.iloc[:, 6:]), axis=1).convert_dtypes()
df = df.set_index(['id_enc'])
df = df.stack().reset_index()
df.columns = ['id_enc','d','sales']
# Reduce mem
df = reduce_mem(df)
#%% 5a) Add event and item features to main df
# Add event information
df = df.merge(df_events, how='left', left_on=['d'], right_on=['d'])
# Add item information
df = df.merge(df_itemids[['id','id_enc', 'item_id_enc', 'dept_id_enc', 'cat_id_enc', 'store_id_enc', 'state_id_enc']], how='left', left_on=['id_enc'], right_on=['id_enc'])
# Reduce mem
df = reduce_mem(df)
#%% 5b) Add selling prices, fill nans.
df = df.merge(df_prices, how='left', right_on=['item_id_enc','store_id_enc','wm_yr_wk'], left_on = ['item_id_enc','store_id_enc','wm_yr_wk'])
df['sell_price'] = df['sell_price'].fillna(-10)
df['sell_price_change'] = df['sell_price_change'].fillna(0)
df['sell_price_norm_item'] = df['sell_price_norm_item'].fillna(0)
df['sell_price_norm_dept'] = df['sell_price_norm_dept'].fillna(0)
df['weeks_on_sale'] = df['weeks_on_sale'].fillna(0)
# Drop columns we no longer need
df = df.drop(columns = ['d','wm_yr_wk'])
df = df.sort_values(by=['item_id_enc','store_id_enc','date'])
df = df.reset_index(drop=True)
#%% 5c) Add time indicators
dates = df['date'].dt
df['dayofweek_sin'] = np.sin(dates.dayofweek * (2 * np.pi / 7))
df['dayofweek_cos'] = np.cos(dates.dayofweek * (2 * np.pi / 7))
df['dayofmonth_sin'] = np.sin(dates.day * (2 * np.pi / 31))
df['dayofmonth_cos'] = np.cos(dates.day * (2 * np.pi / 31))
df['weekofyear_sin'] = np.sin(dates.weekofyear * (2 * np.pi / 53))
df['weekofyear_cos'] = np.cos(dates.weekofyear * (2 * np.pi / 53))
df['monthofyear_sin'] = np.sin(dates.month * (2 * np.pi / 12))
df['monthofyear_cos'] = np.cos(dates.month * (2 * np.pi / 12))
df = reduce_mem(df)
#%% 5e) Add lagged target variables & trends
def add_lags(df, lags):
    group = df.groupby(['id_enc'])['sales']
    for lag in lags:
        df['sales_lag'+str(lag)] = group.shift(lag, fill_value=0).astype('int16')
       
    return df

def add_ma(df, windows, lag):
    group = df.groupby(['id_enc'])['sales_lag'+str(lag)]
    for window in windows:
        df['sales_lag'+str(lag)+'_mavg'+str(window)] = group.transform(lambda x: x.rolling(window, min_periods=1).mean()).fillna(0).astype('float32')

    return df

def add_item_trends(df):
    # Add sales trend indicators
    group = df.groupby(['id_enc'])
    groupma7current = group['sales_lag1_mavg7'].shift(0)
    df['sales_short_trend'] =  (1 + groupma7current) / (1 + group['sales_lag1_mavg28'].shift(0)) - 1
    df['sales_long_trend'] = (1 + groupma7current) / (1 + group['sales_lag28_mavg56'].shift(0)) - 1
    df['sales_year_trend'] = (1 + groupma7current) / (1 + group['sales_lag1_mavg7'].shift(364, fill_value=0)) - 1
    df['sales_lywow_trend'] = (1 + group['sales_lag1_mavg7'].shift(357, fill_value=0)) / (1 + group['sales_lag1_mavg7'].shift(364, fill_value=0)) - 1
    return df

def add_product_trends(df, col, group):
    group_ma7 = group.rolling(7, min_periods=1).mean()
    group_ma90 = group.rolling(91, min_periods=1).mean()
    first_index = group_ma7.index.names[0]
    group_ly = group_ma7.groupby(first_index).shift(364, fill_value=0)
    df_temp = pd.DataFrame(index = group_ma7.index)
    df_temp[col+'_long_trend'] = (1 + group_ma7) / (1 + group_ma90) - 1
    df_temp[col+'_year_trend'] = (1 + group_ma7) / (1 + group_ly) - 1
    df = df.merge(df_temp, how='left', right_on=[first_index,'date'], left_on=[first_index,'date'])
     
    return df

# Add lags and moving averages
df = add_lags(df, np.concatenate((np.arange(1, 8), [28], [56], np.array([364])))) # last 7 days, 28 days, 56 days and last year
df = add_ma(df, np.array([7, 28, 56]), 1) # moving averages
df = add_ma(df, np.array([7, 28, 56]), 7) # moving averages
df = add_ma(df, np.array([7, 28, 56]), 28) # moving averages
df = add_item_trends(df)
df = add_product_trends(df, 'sales_item', df.groupby(['item_id_enc','date'])['sales_lag1'].sum(min_count=1))
# Reduce dataset
df = reduce_mem(df)
#%% Change order of features - this is convenient for later on
# Everything up to 'date' is fixed for each day, thereafter is variable
cols =['sales',
 'date',
 'id',
 'id_enc',
 'sales_lag1',
 'sales_lag2',
 'sales_lag3',
 'sales_lag4',
 'sales_lag5',
 'sales_lag6',
 'sales_lag7',
 'sales_lag1_mavg7',
 'sales_lag1_mavg28',
 'sales_lag1_mavg56',
 'sales_lag7_mavg7',
 'sales_lag7_mavg28',
 'sales_lag7_mavg56',
 'sales_short_trend',
 'sales_long_trend',
 'sales_year_trend',
 'sales_item_long_trend',
 'sales_item_year_trend',
 'item_id_enc',
 'dept_id_enc',
 'cat_id_enc',
 'store_id_enc',
 'state_id_enc',
 'snap_CA',
 'snap_TX',
 'snap_WI',
 'event_name_1_enc',
 'event_type_1_enc',
 'event_name_2_enc',
 'event_type_2_enc',
 'sell_price',
 'sell_price_change',
 'sell_price_norm_item',
 'sell_price_norm_dept',
 'weeks_on_sale',
 'dayofweek_sin',
 'dayofweek_cos',
 'dayofmonth_sin',
 'dayofmonth_cos',
 'weekofyear_sin',
 'weekofyear_cos',
 'monthofyear_sin',
 'monthofyear_cos',
 'sales_lag364',
 'sales_lag28_mavg7',
 'sales_lag28_mavg28',
 'sales_lag28_mavg56',
 'sales_lywow_trend',
 'sales_lag28',
 'sales_lag56']

df = df[cols]
#%% 6) Create hdf with the stuff we need to keep
df['id'] = df['id'].astype('O')
df = df.sort_values(by=['store_id_enc','item_id_enc','date'])
df = df.reset_index(drop=True)
filename = 'pgbm/datasets/m5/m5_dataset_products.h5'
store = pd.HDFStore(filename)
df.to_hdf(filename, key='data')
