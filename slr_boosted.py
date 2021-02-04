import statsmodels.api
import pandas
import numpy

def trend_boosted_slr(
        input_endog,
        n_season,
        forecast_length,
        smooth_factor,
        trend_dampening
    ):
    # Number of available time periods
    n_total = len(input_endog)

    # Number of complete seasons
    nk = n_total/n_season

    # Exog
    #ts = list(input_data[input_exog].values)
    ts = input_endog.values

    # List of exog per season
    ts_list = [ts[i:i + n_season] for i in range(0, len(ts), n_season)]

    # List of average exog per season
    ak_list = [sum(ts_list[i])/len(ts_list[i]) for i in range(0, len(ts_list))]

    # List of starting seasonal index for each time period in a season
    s_list = [ts_list[i]/ak_list[i] for i in range(len(ak_list))]

    # logic for incomplete seasons
    if nk < 1.0:
        s_avg = numpy.mean(s_list)
    elif nk >= 1.0:
        s_avg = [numpy.mean([s[i] for s in s_list if len(s) > i]) for i in range(len(max(s_list,key=len)))]

    # Smooth factor applied to list of seasonal averages
    s_avg_s = numpy.array(s_avg) * smooth_factor

    # Regression model
    X = range(0, len(ts))
    X = statsmodels.api.add_constant(X)
    #Changed to take out seasonality from time series before fitting regression
    y = ts/numpy.resize(s_avg_s, len(ts))

    boosted_data = y
    boosted_output = numpy.zeros(len(y))
    boosted_coefs = numpy.zeros(2)
    for i in range(10):
      model = statsmodels.api.OLS(boosted_data, X)
      res = model.fit()
      fitted = model.predict(res.params, X)
      boosted_output =  boosted_output + .2*fitted
      boosted_data = y - boosted_output
      boosted_coefs = boosted_coefs + .2*res.params

    # Trend dampening applied to the slope of the model
    #res.params[1] = res.params[1]

    # Forecasted regression  model
    X_new = range(0, len(ts)+forecast_length)
    X_new = statsmodels.api.add_constant(X_new)
    yhat = model.predict(boosted_coefs, X_new)
    y_pred = yhat * numpy.resize(s_avg_s, len(yhat))

    # forecast in sample
    frc_in = y_pred[:len(ts)]

    # forecast out sample
    frc_out = y_pred[len(ts):]

    return_dictionary = {
                            'model':                    model,
                            'in_sample_forecast':       pandas.Series(frc_in),
                            'out_of_sample_forecast':   pandas.Series(frc_out),
                            'trend':                    yhat
                        }

    return return_dictionary
