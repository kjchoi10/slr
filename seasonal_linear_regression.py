import statsmodels.api
import pandas
import numpy

class slr:
    
    def __init__(self,
                 input_endog,
                 n_season,
                 forecast_length,
                 smooth_factor,
                 trend_dampening,
                 model_type,
                 boost,
                 boost_iter):
                
                self.input_endog = input_endog
                self.n_season = n_season
                self.forecast_length = forecast_length

                if(smooth_factor is None):
                    self.smooth_factor = 1
                else:
                    self.smooth_factor = smooth_factor
                if(trend_dampening is None):
                    self.trend_dampening = 1
                else:
                    self.trend_dampening = trend_dampening              
                if(model_type is None):
                    self.model_type = 'mult'
                else:
                    self.model_type = model_type

                if(boost is None):
                    self.boost = True
                else:
                    self.boost = boost

                if(boost_iter is None):
                    self.boost_iter = 10
                else:
                    self.boost_iter = boost_iter 

                return

    def get_avg_seasonal_idx(
                            self,
                            ts_list,
                            nk,
                            smooth_factor):
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
        s_avg_s = numpy.array(s_avg) * self.smooth_factor

        return(s_avg_s)


    def fit(self):
        # Number of available time periods
        n_total = len(self.input_endog)

        # Number of complete seasons
        nk = n_total/self.n_season

        # Exog
        ts = self.input_endog.values

        # List of exog per season
        ts_list = [ts[i:i + self.n_season] for i in range(0, len(ts), self.n_season)]

        # Calculate seasonal indices for time series
        self.s_avg_s = self.get_avg_seasonal_idx(
                                        ts_list = ts_list,
                                        nk = nk,
                                        smooth_factor = 1.0)

        if(self.model_type == 'add'):

            # Regression model (add)
            X = range(0, len(ts))
            X = statsmodels.api.add_constant(X)
            # Changed to take out seasonality from time series before fitting regression
            y = ts - numpy.resize(self.s_avg_s, len(ts))

            # Baseline seasonal linear regression
            if(self.boost != 'yes'):
                model = statsmodels.api.OLS(y, X)
                res = model.fit()
                coef = res.params

        if(self.model_type == 'mult'):
            
            # Regression model (mult)
            X = range(0, len(ts))
            X = statsmodels.api.add_constant(X)
            # Changed to take out seasonality from time series before fitting regression
            y = ts/numpy.resize(self.s_avg_s, len(ts))
            
            # Baseline seasonal linear regression
            if(self.boost != True):
                model = statsmodels.api.OLS(y, X)
                res = model.fit()
                coef = res.params


        if(self.boost == True):
            # Boosting
            boosted_data = y
            boosted_output = numpy.zeros(len(y))
            coefs = numpy.zeros(2)

            # Boosting Iterations
            for i in range(self.boost_iter):
                model = statsmodels.api.OLS(boosted_data, X)
                res = model.fit()
                fitted = model.predict(res.params, X)
                boosted_output =  boosted_output + .2*fitted
                boosted_data = y - boosted_output
                coefs = coefs + .2*res.params
        self.model = model
        self.coefs = coefs
        # Forecasted regression  model/prediction
        # X_forecast = range(0, len(self.ts)+self.forecast_length)
        # X_forecast = statsmodels.api.add_constant(X_forecast)
        # yhat = model.predict(coefs, X_forecast)
        # why do we want this?
        # y_pred = yhat * numpy.resize(s_avg_s, len(yhat))

        # forecast in sample
        # frc_in = y_pred[:len(ts)]

        # forecast out sample
        # frc_out = y_pred[len(ts):]

        return

    def predict(self):
        # X values to include ts and forecast
        X = range(0, len(self.input_endog.values)+self.forecast_length)

        # Time as the variable in regression
        X = statsmodels.api.add_constant(X)

        # Predicted Y
        yhat = self.model.predict(self.coefs, X)

        # Predict Y adjusted with seasonal index
        y_adj = yhat * numpy.resize(self.s_avg_s, len(yhat))

        # forecast in sample
        frc_in = y_adj[:len(self.input_endog.values)]

        # forecast out sample
        frc_out = y_adj[len(self.input_endog.values):]

        return_dictionary = {
                            'in_sample_forecast':       pandas.Series(frc_in),
                            'out_of_sample_forecast':   pandas.Series(frc_out),
                            'trend':                    yhat
                            }

        return return_dictionary
