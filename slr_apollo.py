"""
Seasonal Linear Regression
"""

import statsmodels.api
import pandas
import numpy

def fit_slr(
                    data_name,
                    model_time_series_required_length,
                    input_endog,
                    input_dates,
                    input_length,
                    forecast_length,
                    time_grain,
                    input_endog_shifted,
                    forecast_shifted_response,
                    error_logger,
                    training_length_in_years,
                    time_series_class,
                    holidays,
                    training_exog_var,
                    forecast_exog_var
              ):
        """
        :param data_name:
        :param model_time_series_required_length:
        :param input_endog:
        :param input_dates:
        :param input_length:
        :param forecast_length:
        :param time_grain:
        :param input_endog_shifted:
        :param forecast_shifted_response:
        :param error_logger:
        :param training_length_in_years:
        :param time_series_class:
        :param holidays:
        :param training_exog_var:
        :param forecast_exog_var:
        :return:
        """
        if time_series_class == 'near_disco':

                if time_grain == 'week':
                    seasonal_periods = 52
                elif time_grain == 'month':
                    seasonal_periods = 12
                try:

                    slr_training_result = slr(
                                                input_endog = input_endog,
                                                n_season = seasonal_periods,
                                                forecast_length = forecast_length,
                                                smooth_factor = 0.7,
                                                trend_dampening = 0.5)

                    slr_model = slr_training_result['model']
                    slr_fittedvalues = slr_training_result['in_sample_forecast']
                    slr_forecast = slr_training_result['out_of_sample_forecast']

                except Exception as e:

                    slr_model = None
                    slr_fittedvalues = None
                    slr_forecast = None
                    error_logger.error('error in model fit for ' + slr_model + ' ' + ' '.join(data_name) + ' with error ' + str(e))

        else:

            slr_model = None
            slr_fittedvalues = None
            slr_forecast = None


        return slr_model, slr_fittedvalues, slr_forecast

def slr(
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
    s_avg_s = s_avg * smooth_factor

    # Regression model
    X = range(0, len(ts))
    X = statsmodels.api.add_constant(X)
    y = ts
    model = statsmodels.api.OLS(y, X)
    res = model.fit()

    # Trend dampening applied to the slope of the model
    res.params[1] = res.params[1]*trend_dampening

    # Forecasted regression  model
    X_new = range(0, len(ts)+forecast_length)
    X_new = statsmodels.api.add_constant(X_new)
    yhat = model.predict(res.params, X_new)
    y_pred = yhat * numpy.resize(s_avg_s, len(yhat))

    # forecast in sample
    frc_in = y_pred[:len(ts)]

    # forecast out sample
    frc_out = y_pred[len(ts):]

    return_dictionary = {
                            'model':                    model,
                            'in_sample_forecast':       pandas.Series(frc_in),
                            'out_of_sample_forecast':   pandas.Series(frc_out)
                        }

    return return_dictionary
