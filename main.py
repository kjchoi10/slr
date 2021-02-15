import numpy as np
import pandas as pd
import quandl
import matplotlib.pyplot as plt
import seasonal_linear_regression
data = quandl.get("BITSTAMP/USD")

y = data['Low']
y = y[-730:]
    
    
df = pd.DataFrame(y)
df['ds'] = y.index
#adjust to make ready for Prophet
df.columns = ['y', 'ds']

model = seasonal_linear_regression.slr(
        input_endog = df['y'],
        n_season = 12,
        forecast_length = 150,
        smooth_factor = 1,
        trend_dampening = 1,
        model_type = 'mult',
        boost = True,
        boost_iter = 5
    )

boosted_slr = model.fit()

model = boosted_slr['model']
in_sample = boosted_slr['in_sample_forecast']
out_sample = boosted_slr['out_of_sample_forecast']
trend = boosted_slr['trend']

plt.plot(np.append(in_sample, out_sample), label = 'slr_boosted')
plt.plot(trend, label = 'trend')
plt.plot(y.values, color = 'black', label = 'actual')
plt.legend()
plt.show()
