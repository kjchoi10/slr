import numpy as np
import pandas as pd
import quandl
import matplotlib.pyplot as plt
import slr_boosted
data = quandl.get("BITSTAMP/USD")

y = data['Low']
y = y[-730:]
    
    
df = pd.DataFrame(y)
df['ds'] = y.index
#adjust to make ready for Prophet
df.columns = ['y', 'ds']

boosted_slr_obj = slr_boosted.trend_boosted_slr(
        input_endog = df['y'],
        n_season = 12,
        forecast_length = 12,
        smooth_factor = 0.5,
        trend_dampening = 0.5
    )

model = boosted_slr_obj['model']
in_sample = boosted_slr_obj['in_sample_forecast']
out_sample = boosted_slr_obj['out_of_sample_forecast']
trend = boosted_slr_obj['trend']

plt.plot(np.append(in_sample, out_sample), label = 'slr_boosted')
plt.plot(trend, label = 'trend')
plt.plot(y.values, color = 'black', label = 'actual')
plt.legend()
plt.show()
