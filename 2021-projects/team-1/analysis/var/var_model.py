'''
Purpose: Fit a VAR model for comparison with the deep learning models. Plot the VAR predicted vs actual ice extent values.
Source: Code based on https://www.machinelearningplus.com/vector-autoregression-examples-python/
'''
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load train and test data.
df_train = pd.read_csv("/umbc/xfs1/cybertrn/reu2021/team1/research/analysis/var_final_train.csv", index_col=0)
df_test = pd.read_csv("/umbc/xfs1/cybertrn/reu2021/team1/research/analysis/var_final_test.csv", index_col=0)

# Define column names.
df_train.columns = ["sp", "wind", "humidity", "temp", "shortwave", "longwave", "rain", "snow", "sst", "sea_ice", "ice_extent"]
df_test.columns = ["sp", "wind", "humidity", "temp", "shortwave", "longwave", "rain", "snow", "sst", "sea_ice", "ice_extent"]

nobs = 96

# ADFuller Test
for name, col in df_train.iteritems():
  result = adfuller(col.dropna())
  print(f'{name} ADF Statistic: {result[0]}')
  print(f'{name} p-value: {result[1]}')
  print("=========================")

# Determine correct order of lag
model = VAR(df_train)
for i in [1,2,3,4,5,6]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')
    print("=========================")

x = model.select_order(maxlags=12)
print(x.summary())

# Model fitting based on optimal lag. A lag of two was chosen for this model.
model_fitted = model.fit(2)
print(model_fitted.summary())

# Check for Serial Correlation.
from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(df_train.columns, out):
    print(col, ':', round(val, 2))

# Forecasting
forecast_input = df_test.values
fc = model_fitted.forecast(y=forecast_input, steps=nobs) # Obtain forecasts for each variable for 2013-2020
df_forecast = pd.DataFrame(fc, index=df_test.index, columns=df_test.columns + '_2d')
df_forecast.columns = [i + "_forecast" for i in df_test.columns]

# Save predictions and real values to numpy arrays.
with open("var_forecast_2013-2020.npy", "wb") as f:
	np.save(f, df_forecast['ice_extent_forecast'].to_numpy())
with open("var_real_2013-2020.npy", "wb") as f:
	np.save(f, df_test['ice_extent'].to_numpy())

from sklearn.metrics import mean_squared_error
import math
print(f"VAR Test RMSE: {math.sqrt(mean_squared_error(df_test['ice_extent'], df_forecast['ice_extent_forecast']))}")

# Forecast Plotting
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(10, 6))
df_test["ice_extent"].plot(legend=True, ax=ax)
df_forecast["ice_extent_forecast"].plot(legend=True, ax=ax).autoscale(axis="x", tight=True)
ax.set_title("Ice Extent (million km^2) : Forecast vs Actual 2013-2020", fontsize=18)
ax.xaxis.set_ticks(range(0, 96, 12))
ax.xaxis.set_ticklabels([])
plt.tight_layout(pad=4.0, h_pad = 4.0, w_pad = 4.0)

fig.savefig("var_forecast_vs_actual_extent_2013-2020.png")
