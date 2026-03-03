import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error

plt.style.use("fivethirtyeight")
color_pal = sns.color_palette()

df = pd.read_csv("/content/PJME_hourly.csv")
df = df.set_index("Datetime")
df.index = pd.to_datetime(df.index)

# visualize data
df.plot(style = ".",
        figsize = (15, 8),
        title = "PJME Enegy Use in MW",
        color = color_pal[0])
plt.show()

################################################################Outlier Analysis and removal#################################################
df["PJME_MW"].plot(kind='hist', bins=500)

# Checking where Outliers are
df.query("PJME_MW < 20000").plot(figsize=(15,5), style='.', color=color_pal[3], title="Outlier Analysis [below 20000 MW]")

df.query('PJME_MW > 19000').plot(figsize=(15, 5), style = ".")

df = df.query('PJME_MW > 19000').copy()

# Reviewing: Train / Test Split
train = df.loc[df.index < "01-01-2015"]
test = df.loc[df.index >= "01-01-2015"]

fig, ax = plt.subplots(figsize = (15, 5))
train.plot(ax=ax, label="Train Set",
           title = "Data Train / Test Split")
test.plot(ax=ax, label="Test Set")
ax.axvline("01-01-2015", color="black", ls="--")
ax.legend(["Train Set", "Test Set"])
plt.show()

# visualize
df.plot(style = ".",
        figsize = (15, 5),
        color = color_pal[0],
        title = "PJME Energy Use in MW")
plt.show()

# Outlier Analysis and removal
df['PJME_MW'].plot(kind="hist", bins=500)

df.query('PJME_MW < 20000').plot(figsize=(15, 5), style=".")

df = df.query('PJME_MW > 19000').copy()

################################################ Reviewing: Train/ Test Split [basic Method]
train = df.loc[df.index < "01-01-2015"]
test = df.loc[df.index >= "01-01-2015"]

fig, ax = plt.subplots(figsize=(15, 5))

train.plot(ax=ax, label="train Set",
           title = "Data Train/Test Split")
test.plot(ax=ax, label="Test Set")
ax.axvline("01-01-2015", color="black", ls="--")
ax.legend(["Training Set", "Test Set"])
plt.show()

####################################################################### More robust time series cross-validation######################################
from sklearn.model_selection import TimeSeriesSplit
tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)

df = df.sort_index()

# Visualize
fig, axs = plt.subplots(5, 1,figsize=(17,9), sharex = True)
fold = 0

for train_idx, val_idx in tss.split(df):
  train = df.iloc[train_idx]
  test = df.iloc[val_idx]
  train['PJME_MW'].plot(ax=axs[fold],
                        label="Training Set",
                        title = f"Data Train/Test Split Fold{fold}")
  test['PJME_MW'].plot(ax=axs[fold], label="Test Set")
  axs[fold].axvline(test.index.min(), color="black", ls="--")
  fold += 1

plt.tight_layout()
plt.show()

########################################################### Forecasting Horizon Explained#######################################################

def creative_features(df):
  df = df.copy()
  df["hour"] = df.index.hour
  df["dayofweek"] = df.index.dayofweek
  df["quarter"] = df.index.quarter
  df["month"] = df.index.month
  df["year"] = df.index.year
  df["dayofyear"] = df.index.dayofyear
  df["dayofmonth"] = df.index.day
  df["weekofyear"] = df.index.isocalendar().week
  return df

df = creative_features(df)

# plot for year 2016
ig, ax = plt.subplots(figsize=(15, 6))

year = 2016
df[df.index.year == year]['PJME_MW'].plot(ax=ax,
                                          label=str(year),
                                          alpha=1,
                                          ms=1,
                                          lw=1)

# Format x-axis to show month names
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())

# Set title with the year
plt.title(f'PJME_MW - {year}', fontsize=14, fontweight='bold')

# Remove the default x-label
ax.set_xlabel('')

plt.tight_layout()
plt.show()

# Last 3 years plot to see pattern (excluding 2018) as the prediction is just for 2019
fig, ax = plt.subplots(figsize=(15, 6))

for year in [2015, 2016, 2017]:
    df[df.index.year == year]['PJME_MW'].plot(ax=ax,
                                               label=f'Year {year}',
                                               alpha=0.7)

plt.title('PJME_MW - Multiple Years Comparison')
plt.legend()
plt.show()

########################## Adding lag features
def add_lags(df):
  target_map = df['PJME_MW'].to_dict()
  df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
  df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
  df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
  return df

df = add_lags(df)

######################################### Training Using Cross-validation######################################################################

tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df.sort_index()
fold = 0
preds = []
scores = []

for train_idx, val_idx in tss.split(df):
  train = df.iloc[train_idx]
  test = df.iloc[val_idx]

  train = creative_features(train)
  test = creative_features(test)

  FEATURES = ["dayofyear", "hour", "dayofweek", "quarter", "month", "year", "lag1", "lag2", "lag3"]

  TARGET = 'PJME_MW'

  X_train = train[FEATURES]
  y_train = train[TARGET]

  X_test = test[FEATURES]
  y_test = test[TARGET]

  reg = xgb.XGBRegressor(base_score=0.5,
                         booster='gbtree',
                         n_estimators=1000,
                         early_stopping_rounds=50,
                         objective='reg:linear',
                         max_depth=3,
                         learning_rate=0.01
                         )

  reg.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          verbose=100)

  y_pred = reg.predict(X_test)
  preds.append(y_pred)
  score = np.sqrt(mean_squared_error(y_test, y_pred))
  scores.append(score)
  print(f'Score across folds {np.mean(scores)}')
  print(f'folds scores {scores}')

########################## Retraining the model to predict the future####################################################
# Retrain all data
df = creative_features(df)

FEATURES = ["dayofweek", "hour", "dayofweek", "quarter", "month", "year", "lag1", "lag2", "lag2", "lag3 "]

TARGET = 'PJME_MW'

X_all = df[FEATURES]
y_all = df[TARGET]

reg = Xgb.XGBRegressor(base_score=0.5,
                       booster=gbtree,
                       n_estimators=500,
                       objective='reg: linear',
                       max_depth=3,
                       learning_rate=0.01)

reg.fit(X_all, y_all,
        eval_set=[(X_all, y_all)],
        verbose=100)

df.index_max()

################################################## Create future dataframe
future = pd.date_range('2018-08-03', '2019-08-01', freq='1h')
future_df = pd.DataFrame(index=future)
future_df['isFuture'] = True
df['isFuture'] = False
df_and_future = pd.concat([df, future_df])

df_and_future = creative_features(df_and_future)
df_and_future = add_lags(df_and_future)

######## Adding features to the future data frame
future_with_features = df_and_future.query('isFuture').copy()
future_with_features

############################################################## Predicting the future#############################################
future_with_features['pred'] = reg.predict(future_with_features[FEATURES])
future_with_features

######### visualising the future dataframe
future_with_features['pred'].plot(figsize=(17, 9),
                          color=color_pal[3],
                          ms=1,
                          lw=1,
                          title='Future Predictions')

plt.show()

################################################################# Saving the model and loading the model #################################
reg.save_model('model.json')
!ls - lh
############### visual verification that we get the same output from the saved model
reg_new = xgb.XGBRegressor()
reg_new.load_model('model.json')
future_with_features['pred'] = reg.predict(future_with_features[FEATURES])
future_with_features['pred'].plot(figsize=(17, 9),
                          color=color_pal[5],
                          ms=1,
                          lw=1,
                          title='Future Predictions',
                          legend=False)
plt.show()
