
import statsmodels.api as sm
import pandas as pd
import numpy as np

def slr(
        input_data,
        input_endog,
        input_exog,
        n_season,
        forecast_length,
        smooth_factor,
        trend_dampening
    ):
    # Number of available time periods
    n_total = len(input_data)

    # Number of complete seasons
    nk = n_total/n_season

    # Exog
    ts = list(input_data[input_exog].values)

    # List of exog per season
    ts_list = [ts[i:i + n_season] for i in range(0, len(ts), n_season)]

    # List of average exog per season
    ak_list = [sum(ts_list[i])/len(ts_list[i]) for i in range(0, len(ts_list))]

    # List of starting seasonal index for each time period in a season
    s_list = [ts_list[i]/ak_list[i] for i in range(len(ak_list))]

    # logic for incomplete seasons
    if nk < 1.0:
        s_avg = np.mean(s_list)
    elif nk >= 1.0:
        s_avg = [np.mean([s[i] for s in s_list if len(s) > i]) for i in range(len(max(s_list,key=len)))]

    # Smooth factor applied to list of seasonal averages
    s_avg_s = s_avg * smooth_factor

    # Regression model
    X = range(0, len(ts))
    X = sm.add_constant(X)
    y = ts
    model = sm.OLS(y, X)
    res = model.fit()

    # Trend dampening applied to the slope of the model
    res.params[1] = res.params[1]*trend_dampening

    # Forecasted regression  model
    X_new = range(0, len(ts)+forecast_length)
    X_new = sm.add_constant(X_new)
    yhat = model.predict(res.params, X_new)
    y_pred = yhat * np.resize(s_avg, len(yhat))

    return ts, X_new, yhat, y_pred
