
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
    n_total = len(input_data)
    nk = n_total/n_season
    smooth_factor = 1

    ts = list(input_data[input_exog].values)

    ts_list = [ts[i:i + n_season] for i in range(0, len(ts), n_season)]

    ak_list = [sum(ts_list[i])/len(ts_list[i]) for i in range(0, len(ts_list))]

    s_list = [ts_list[i]/ak_list[i] for i in range(len(ak_list))]

    s_avg = [np.mean([s[i] for s in s_list if len(s) > i]) for i in range(len(max(s_list,key=len)))]

    s_avg_s = s_avg * smooth_factor

    ts_new = ts * np.resize(s_avg_s, len(ts))

    trend_dampening = 1.0

    res.params[1] = res.params[1]*trend_dampening

    X = range(0, len(ts_new))
    X = sm.add_constant(X)
    y = ts_new

    model = sm.OLS(y, X)
    res = model.fit()

    X_new = range(0, len(ts_new)+10)
    X_new = sm.add_constant(X_new)
    yhat = model.predict(res.params, X_new)
    y_pred = yhat * np.resize(s_avg, len(yhat))

    return ts_new, X_new, yhat, y_pred
