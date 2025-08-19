import kagglehub
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')


# 1. Load data
df = pd.read_csv(r'H:\learn\archive\PJME_hourly.csv')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)
print(df)

# 2. Plots

# 2.1.1 Overall

df.plot(style='.',
        figsize=(15,5),
        color=color_pal[0],
        title='PJME Energy Use in MW')
plt.show()

# 2.1.2 
#df['PJME_MW'].plot(kind='hist', bins=700)
#plt.show()

# 3. Outlier Analysis
df.query('PJME_MW < 19000').plot(figsize=(15,5), style='.')
plt.show()

# Chosen data
df = df.query('PJME_MW > 19000').copy()
# plotting it
df.query('PJME_MW > 19000').plot(figsize=(15,5))

# 4.1 Simpler version of splitting data

train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set',
           title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

# 4.2 Appropriate splitting option

tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()

# plotting subplots

fig, axs = plt.subplots(5, 1, figsize=(15,5), sharex=True)

fold = 0
for train_idx, val_idx in tss.split(df):
  train = df.iloc[train_idx]
  test = df.iloc[val_idx]
  train['PJME_MW'].plot(ax=axs[fold], label='Training Set',
                        title=f'Data Train/Test Split Fold {fold}')
  test['PJME_MW'].plot(ax=axs[fold], label='Test Set')
  #ax.axvline(test.index.min(), color='black', ls='--')
  fold += 1
plt.show()

#----------------------------------------------------------------FORECASTING HORIZON------------------------------------------------------------------------
def create_features(df):
  df = df.copy()
  df['hour'] = df.index.hour
  df['dayofweek'] = df.index.dayofweek
  df['quarter'] = df.index.quarter
  df['month'] = df.index.month
  df['year'] = df.index.year
  df['dayofyear'] = df.index.dayofyear
  df['dayofmonth'] = df.index.dayofmonth
  df['weekofyear'] = df.index.isocalender().week
  return df
