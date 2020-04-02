# model using statsmodel
import statsmodels.api as sm
from itertools import cycle
import pandas as pd
import numpy as np
import datetime as dt

def fit_slr()


def _slr(
            input_data,
            input_endog,
            input_exog,
            input_season,
            forecast_length,
        ):
   """
   :param input_data: pandas dataframe of input_endog and input_exog
   :param input_endog: string name of input_endog
   :param input_exog: string name of input_exog
   :param input_season: integer for number of periods in a season
   :param forecast_length: integer of forecast length
   :return:
   """

   input_data['period_number'] = [t.month for t in input_data[input_endog]]

   n_total = len(input_data)

   # seasonal moving average
   m_a = input_data[input_exog].rolling(window=input_season).mean()

   # centered moving average
   m_a_centered = m_a.rolling(window=2).mean().dropna().tolist()
   midpoint = (input_season/2)+1

   # add centered moving average starting at midpoint index
   input_data.loc[input_data.index[int(midpoint): int(midpoint) + len(m_a_centered)], 'm_a_centered'] = m_a_centered

   # season irregular value
   input_data['season_irregular'] = input_data[input_exog]/input_data['m_a_centered']

   # calculate average seasonal index by period - fill NAN season_irregular with 1 (trend)
   seasonal_index = input_data.groupby('period_number').agg({'season_irregular': 'mean'}).fillna(1).reset_index().rename({'season_irregular': 'season_index'}, axis=1)

   # merge seasonal_index with output_data
   output_data = pd.merge(input_data, seasonal_index, on=['period_number'])

   

   return()


def fit_slr(df):
  """
  :params: pandas time series with seasonal index calculated
  :return: model object, forecast
  """

  # create year column
  df['month_number'] = [x.month for x in df['month']]

  # total number of months
  n_total = len(df)

  # number of months in a season
  n_season = 12

  # moving average
  m_a = df['training_response'].rolling(window=n_season).mean()

  # centered moving average
  m_a_centered = m_a.rolling(window=2).mean().dropna().tolist()
  midpoint = (n_season/2)+1

  # add centered moving average starting at midpoint index
  df.loc[df.index[int(midpoint):int(midpoint)+len(m_a_centered)], 'm_a_centered'] = m_a_centered

  # seasonal irregular value
  df['season_irregular'] = df['training_response']/df['m_a_centered']

  # extend the rest of the season with the latest m_a value / or for now we can just replace NAN values with 1 which is trend
  seasonal_index = df.groupby('month_number').agg({'season_irregular': 'mean'}).fillna(1).reset_index().rename({'season_irregular': 'seasonal_index'}, axis = 1)
  d = pd.merge(df, seasonal_index, on = ['month_number'])

  # apply seasonal index to soh
  d['coef'] = d['training_response'] * d['seasonal_index']

  # sort months
  d = d.sort_values(by='month')

  # period calculation
  d['period'] = range(len(d))

  d = d.reset_index()

  # regression
  df = d

  X = df['period']
  X = sm.add_constant(X)
  y = df['training_response']

  model = sm.OLS(y, X)
  res = model.fit()

  # prediction input (period)
  X_lately = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21, 22, 23, 24, 25, 26, 27]
  # intercept
  constant = np.ones_like(X_lately)
  # create prediction input
  X_new = np.column_stack((constant, X_lately))
  # list of forecasts including in-sample and out-sample
  forecast_set = model.predict(res.params, X_new)
  # forecast dataframe
  df_forecast = pd.DataFrame({'period': X_lately, 'forecast': forecast_set})

  # Adjusted period for repeating 0-12 months
  seq = cycle(range(13))

  df_forecast['period_adj'] = [next(seq) for count in range(df_forecast.shape[0])]

  # Right join seasonal index
  df_forecast = df_forecast.merge(df[['seasonal_index', 'period']], left_on='period_adj', right_on='period', how='left')

  # Forecast adjusted based on the period and seasonal index
  df_forecast['forecast_adj'] = df_forecast['forecast'] * df_forecast['seasonal_index']

  return(model, df_forecast)
