# Databricks notebook source
# MAGIC %md
# MAGIC # Purpose/Problem Statement
# MAGIC This purpose of this part of the analysis is to build a model to forcast sales over the next 7 days on the hourly level (168 forecasted values). We only want to use the bike count (rental history) alone for this problem.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Generate Cleaned Dataset

# COMMAND ----------

# MAGIC %run ./0_data_preprocessing_seoul_bike_project

# COMMAND ----------

# MAGIC %md
# MAGIC #### Package Imports

# COMMAND ----------

# MAGIC %pip install pmdarima
# MAGIC %pip install prophet

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sps
import statsmodels.api as sm
import statsmodels.formula.api as smf
from  statsmodels.tsa.statespace.sarimax import SARIMAX
from  statsmodels.tsa.arima.model import  ARIMA

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error

import plotly.express as px
import plotly.figure_factory as ff

from plotly import subplots
import warnings


%matplotlib inline
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion & Inspection
# MAGIC Read in the cleaned dataset and filter it to only those records that have a valid bike rental count (to reduce noise).

# COMMAND ----------

#read in the cleaned parquet file, which was created in a previous notebook
df_raw = pd.read_parquet('seoul_bike_dataset_final')

# COMMAND ----------

df_raw.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspection of rental bikes data by date and time

# COMMAND ----------

fig, axs= plt.subplots(figsize=(20, 8), nrows=2)
sns.lineplot(x = 'date', y = 'rented_bike_count', data = df_raw.groupby(['date'], as_index=False)['rented_bike_count'].sum(), ax=axs[0]).set(title = "Daily accumulated rental bike count")
sns.lineplot(x = 'date', y = 'rented_bike_count', data = df_raw, hue='hour', ax=axs[1]).set(title = "Daily rental bike count by hour")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Findings/Notes: <br>
# MAGIC We know that there is a flag in the dataset called 'functioning_day' which may have an impact on the overall number of rentals, specifically:
# MAGIC - If the rental agency wasn't functioning on a specific day, there couldn't be bikes rented.
# MAGIC - These timepoints may not reflect a typical pattern of rentals, so we should consider whether we want our timeseries to actually learn them or not. 
# MAGIC 
# MAGIC We can look at the data to see what it looks like and make a determination.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature creation/transformations

# COMMAND ----------

#examine the functioning day variable - is it useful in our analysis?
df_raw.groupby('functioning_day')['rented_bike_count'].agg(['count','mean', 'min', 'max'])

# COMMAND ----------

df_raw.loc[df_raw['functioning_day'] == 'No', 'date'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC There are 12 non-functioing days and one day (10/6/2018) where 7 hours were non-functioning. 
# MAGIC 
# MAGIC ASSUMPTION TO CHECK: We assume that these non-functioning days aren't something we want the model to learn to forecast, so we need to deal with these values. Let's look at them.

# COMMAND ----------

fig, axs= plt.subplots(figsize=(10, 8))
sns.lineplot(x = 'date_time', y = 'rented_bike_count', data = df_raw.loc[df_raw['date'].isin(['2018-09-29','2018-09-30','2018-10-01','2018-10-02','2018-10-03','2018-10-04','2018-10-05','2018-10-06'])])

# COMMAND ----------

# MAGIC %md
# MAGIC Instead of using 0s, lets impute these values. We can always take this out later if we learn that our assumption was incorrect.

# COMMAND ----------

#replace 0s with missings so we can use interpolation
df_final = df_raw.copy()
#first replace the 0s with missing so we can use the interpolate method
df_final.loc[df_final['functioning_day'] =='No', 'rented_bike_count'] = df_final['rented_bike_count'].replace(0, np.nan)
df_final['imputed'] = df_final['rented_bike_count'].isna()
df_final['rented_bike_count'] = df_final['rented_bike_count'].interpolate(method = 'linear')
df_final.loc[df_final['date'].isin(['2018-09-29','2018-09-30','2018-10-01'])]

# COMMAND ----------

fig, axs= plt.subplots(figsize=(10, 8))
sns.scatterplot(x = 'date_time', y = 'rented_bike_count', style='imputed', data = df_final.loc[df_final['date'].isin(['2018-09-29', '2018-09-30','2018-10-01','2018-10-02','2018-10-03','2018-10-04','2018-10-05','2018-10-06'])])

# COMMAND ----------

# MAGIC %md
# MAGIC # Modeling for Forecasting
# MAGIC 
# MAGIC Forecasting bike_rental_count using just historical bike_rental_count. That is bike_rental_counts from previous days/hours. We need 7 days worth at the hour level, 168 days.
# MAGIC 
# MAGIC Reminder of the nuts and bolts of time series models:
# MAGIC  - Trend = Overall increases, decreases, or remains constant over time.
# MAGIC  - Seasonal/periodic = Patterns that repeat over time via periods, like seasons. 
# MAGIC  - Random/Noise = Variableility in the data that cannot be explained by the model.

# COMMAND ----------

df_ts = df_final[['date_time','rented_bike_count']]
df_ts.set_index('date_time', inplace=True)
df_ts.info()

# COMMAND ----------

#seasonal decomposition, we use multiplicable here because the trend and seasonal variation is increasing/decreasing over time. If it were linear we would have used addditive.
decomposition = sm.tsa.seasonal_decompose(df_ts['rented_bike_count'], model='multiplicative')
decomposition.plot()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create train/test splits of data
# MAGIC Hold out the last month of data for final model evaluation.

# COMMAND ----------

train = df_ts[df_ts.index < pd.to_datetime("2018-11-01", format = '%Y-%m-%d')]
test = df_ts[df_ts.index >= pd.to_datetime("2018-11-01", format = '%Y-%m-%d')]
print(train.shape, test.shape)

# COMMAND ----------

#view splits of data
plt.plot(train, color = "blue")
plt.plot(test, color = "green")
plt.ylabel('Bike Rentals (count)')
plt.xlabel('Date/hour')
plt.xticks(rotation=45)
plt.title("Train/Test partitioning")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## auto-ARIMA

# COMMAND ----------

import pmdarima as pm

# COMMAND ----------

# Fit model
model = pm.auto_arima(train, seasonal=True, m=12)
# make forecasts
forecasts = model.predict(test.shape[0])  # predict N steps into the future

# COMMAND ----------

# MAGIC %md
# MAGIC code to 
# MAGIC serialize model with pickle
# MAGIC with open('arima.pkl', 'wb') as pkl:
# MAGIC     pickle.dump(model, pkl)
# MAGIC     
# MAGIC     

# COMMAND ----------

model

# COMMAND ----------

# Visualize the forecasts (blue=train, green=forecasts)
plt.plot(train.index, train['rented_bike_count'], c='blue')
plt.plot(test.index, forecasts, c='green')
plt.axhline(y=df_ts['rented_bike_count'].mean(), color = 'r')
plt.show()

# COMMAND ----------

# Visualize the forecasts compared to the actual values (blue=train, green=forecasts)
plt.plot(df_ts.index, df_ts['rented_bike_count'], c='blue')
plt.plot(test.index, forecasts, c='green')
plt.axhline(y=df_ts['rented_bike_count'].mean(), color = 'r')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Not looking so great here, maybe at first, but then it's become too regularized.

# COMMAND ----------

# MAGIC %md
# MAGIC ## SARIMA
# MAGIC Since we know we have a seasonal component here, let's try the SARIMA model
# MAGIC - (AR) = autoregression, element is aka p, order is P
# MAGIC - (i) = differencing, element is aka d, order is D
# MAGIC - (MA) = moving average, element is aka q, order is D
# MAGIC - + seasonal component for the period of seasonality, element is aka m, the number of time steps for a single seasonal period.
# MAGIC 
# MAGIC Model is specified as SARIMA(p,d,q)(P,D,Q)m
# MAGIC 
# MAGIC Our final model will be specified according to the hyperparameters we chose above:
# MAGIC 
# MAGIC SARIMA(order=(2, 1, 0), scoring_args={}, seasonal_order=(2, 0, 0, 12),
# MAGIC       suppress_warnings=True, with_intercept=False)
# MAGIC 
# MAGIC Notes:
# MAGIC - For example, p=1 represents the first seasonailtiy offeset e.g. t-(m*1) or t-12. P=2 would use the last 2 seasonally offset observations t-(m*1), t-(m*2)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's evaluate the cross validation score for the SARIMA model we chose using the entire dataset.

# COMMAND ----------

from sklearn.model_selection import TimeSeriesSplit

# COMMAND ----------

warnings.filterwarnings("ignore")

tscv = TimeSeriesSplit(n_splits = 4)
rmse = []
for train_index, test_index in tscv.split(df_ts):
    cv_train, cv_test = df_ts.iloc[train_index], df_ts.iloc[test_index]
    ARMA_model = SARIMAX(cv_train, order= (2,1,0), seasonality_order = (2,0,0,12)).fit(disp=False)
    
    predictions = ARMA_model.predict(cv_test.index.values[0], cv_test.index.values[-1])
    true_values = cv_test.values
    rmse.append(np.sqrt(mean_squared_error(true_values, predictions)))
    
print("RMSE: {}".format(np.mean(rmse)))

# COMMAND ----------

# MAGIC %md
# MAGIC Our 4 fold RMSE is 684 bikes.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict values for the next 7 days

# COMMAND ----------

from pandas import datetime
ARMA_model = SARIMAX(df_ts, order= (2,1,0), seasonality_order = (2,0,0,12)).fit(disp=False)

start_index = datetime(2018, 12, 1,0)
end_index = datetime(2018, 12, 7,23)

final_forecasted_values = ARMA_model.predict(start = start_index, end=end_index)
final_forecasted_values.head()

# COMMAND ----------

fig, axs= plt.subplots(figsize=(20, 10))
plt.plot(df_ts.index, df_ts['rented_bike_count'], c='blue')
plt.plot(final_forecasted_values.index, final_forecasted_values, c='r')
plt.axhline(y=df_ts['rented_bike_count'].mean(), color = 'green')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prophet
# MAGIC Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. In other words, it frames the forecasting problem as though it is a curve-fitting exercise.
# MAGIC 
# MAGIC Prophet decomposes the time series (like ARIMA) into trend (in this case linear, non-periodic), seasonality (periodic, such as weekly, monthly, yearly that an expotential smoothing function might apply), and holidays (irregular schedules).
# MAGIC * Works best with time series that have strong seasonal effects and several seasons of historical data. 
# MAGIC * Robust to missing data and shifts in the trend.
# MAGIC * Typically handles outliers well.
# MAGIC * Can be tuned, parameters are: 1. growth (the form of trend, linear or logistic), 2. changepoint_range: the range determines how close the changepoints can go to the end of the time series (larger = more flexible) 3.-100. potential seasons, among other things.
# MAGIC * Assumptions: error term is normally distributed.

# COMMAND ----------

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly, plot_cross_validation_metric

# COMMAND ----------

df_prophet =df_ts.reset_index()
df_prophet.rename(columns = {'date_time': 'ds','rented_bike_count':'y'}, inplace=True)
df_prophet.head()

# COMMAND ----------

prophet_model = Prophet()
prophet_model.fit(df_prophet)

# COMMAND ----------

from prophet.serialize import model_to_json, model_from_json

with open('serialized_model.json', 'w') as fout:
    fout.write(model_to_json(prophet_model))  # Save model

with open('serialized_model.json', 'r') as fin:
    prophet_model = model_from_json(fin.read())  # Load model

# COMMAND ----------

#predict future values
predicted_periods = prophet_model.make_future_dataframe(periods = 168)
forecast = prophet_model.predict(predicted_periods)

# COMMAND ----------

plot_plotly(prophet_model, forecast)

# COMMAND ----------

# MAGIC %md
# MAGIC This figure shows the general forecast of the model, the black dots representing the data points (actual) and the bands representing the distribution of predictions. 

# COMMAND ----------

plot_components_plotly(prophet_model, forecast)

# COMMAND ----------

# MAGIC %md
# MAGIC Diagnose the model using cross validation.

# COMMAND ----------

from prophet.diagnostics import cross_validation, performance_metrics

# COMMAND ----------

df_cv = cross_validation(prophet_model, initial = '120 days', period = '30 days', horizon = '90 days' )

# COMMAND ----------

# MAGIC %md
# MAGIC Compute performance metrics on the predictions.

# COMMAND ----------

df_p = performance_metrics(df_cv)
df_p.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize the performance metrics

# COMMAND ----------

fig = plot_cross_validation_metric(df_cv, metric = 'rmse')

# COMMAND ----------

# MAGIC %md
# MAGIC As expected, the rmse goes up as the horizon gets larger. Generally speaking we are in the 500 to 600 rmse range, meaning on average our predictions fall within these values from the actual values. Not great.

# COMMAND ----------

# MAGIC %md
# MAGIC Next steps would involve parameter tuning, including additional season terms for the calendar seasons.
# MAGIC 
# MAGIC 
# MAGIC https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html is a useful resource for this
# MAGIC 
# MAGIC This requires some additional dataset finessing (more dev time) so this will be pushed off as next steps.

# COMMAND ----------

# MAGIC %md
# MAGIC # Final thoughts/conclusion:
# MAGIC While neither model is particulary great, the prophet model shows promise:
# MAGIC  
# MAGIC Trend:
# MAGIC - between the months of February through July there is an increasing trend
# MAGIC - slight dip in autum, with an uptick in October
# MAGIC - General decline after October though January - although this covers the period of the 2018 winter olympics, would need more context.
# MAGIC   
# MAGIC 'Seasons':
# MAGIC - Sundays have least sales, weekdays are mostly constant and Saturdays are highest (combination of those running errands and recreation)
# MAGIC - Two general spikes in hours, around 8am and then again at 6pm. 
# MAGIC 
# MAGIC Suggested next steps:
# MAGIC - Continue to collect data as this will improve both model's performance
# MAGIC - Consider seasonal (quarterly) models (bespoke or other type)
# MAGIC - Continue to tune the models, prophet mostly.
# MAGIC - Determine if Winter values were impacted by the winter olympics
# MAGIC - Consider other models that have better performance, like prophet or an lstm model (lose interpretability)
