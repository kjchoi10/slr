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

   # apply seasonal index to input_exog


   return()

 # model using statsmodel
import statsmodels.api as sm
from itertools import cycle
import numpy as np
import pandas as pd
import math

def slr(df, forecast_length, n_season, trend_dampening):
  """
  :params df: pandas time series with seasonal index calculated
  :params forecast_length: integer for length forecast
  :params n_season: type of season (i.e. 12 for monthly)
  :params trend_damp: float number where 1 is default
  :return: model object, forecast
  """
  # available seasons
  n_k = len(df)/n_season

  # create year column
  df['month_number'] = [x.month for x in df['month']]

  # total number of months
  n_total = len(df)

  # number of months in a season
  n_season = 12

  # moving average
  m_a = df['training_response'].rolling(window=n_season).mean().dropna().tolist()

  # centered moving average
  #m_a_centered = m_a.rolling(window=2).mean().dropna().tolist()

  # (need to add even or odd check)
  midpoint = (n_season/2)+1

  # estimate remaining centered moving average
  if(n_k.is_integer() == False):

    # account for after a period
    trailing = 1

    # tail of m_a
    m_a_remainder = ((math.ceil(n_k)-n_season)-n_total)+trailing

    # total m_a
    m_a_len = len(m_a) + m_a_remainder

    # period for m_a once centered
    period_ma = np.array(range(int(midpoint), int(midpoint)+len(m_a)))

    # forecast period for m_a once centered
    forecast_period_ma = np.array(range(int(midpoint)+len(m_a), n_total))

    # returns tail m_a
    m_a_res = moving_average_estimate(m_a, period_ma, forecast_period_ma)

    # full m_a
    m_a.extend(m_a_res)

  elif(n_k.is_interger() == True):

    m_a = m_a

  # add centered moving average starting at midpoint index
  df.loc[df.index[int(midpoint):int(midpoint)+len(m_a)], 'm_a_centered'] = m_a

  # seasonal irregular value
  df['season_irregular'] = df['training_response']/df['m_a_centered']

  # extend the rest of the season with the latest m_a value / or for now we can just replace NAN values with 1 which is trend
  seasonal_index = df.groupby('month_number').agg({'season_irregular': 'mean'}).fillna(1).reset_index().rename({'season_irregular': 'seasonal_index'}, axis = 1)
  d = pd.merge(df, seasonal_index, on = ['month_number'])

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

  # dampen the beta coefficent
  res.params[1] = res.params[1]*trend_dampening

  # prediction input (period)
  forecast_period = np.array(range(len(df)+forecast_length))

  # intercept
  forecast_constant = np.ones_like(forecast_period)

  # create prediction input
  forecast_input = np.column_stack((forecast_constant, forecast_period))

  # list of forecasts including in-sample and out-sample
  forecast_set = model.predict(res.params, forecast_input)

  # forecast dataframe
  forecast_output = pd.DataFrame({'period': forecast_period, 'forecast': forecast_set})

  # Adjusted period for repeating 0-12 months
  seq = cycle(range(n_season))

  forecast_output['period_adj'] = [next(seq) for count in range(forecast_output.shape[0])]

  # Right join seasonal index
  forecast_output = forecast_output.merge(df[['seasonal_index', 'period']], left_on='period_adj', right_on='period', how='left')

   # Forecast adjusted based on the period and seasonal index
  forecast_output['forecast_adj'] = forecast_output['forecast'] * forecast_output['seasonal_index']

  return(res, forecast_output, df)


def moving_average_estimate(y, X, forecast_period):
  X = sm.add_constant(X)
  model = sm.OLS(y,X)
  res = model.fit()
  forecast_constant = np.ones_like(forecast_period)
  forecast_input = np.column_stack((forecast_constant, forecast_period))
  forecast_set = model.predict(res.params, forecast_input)
  return(forecast_set)
