# Databricks notebook source
# MAGIC %md
# MAGIC #### Purpose/Context
# MAGIC At this point, we have performed data processing and cleaning in the '0 data preprocessing seoul bike project'  workbook. This part of the analysis will focus on exploring the data.
# MAGIC 
# MAGIC The data spans the period of December 1, 2017 to November 30, 2018. A quick google major events in this period reveals that the Winter Olympics took place in this time period. Since this is an event that typically only happens once and results in tremendous anomalies in the city it takes place in, this will pose issues when trying use this data to predict future events. We can dig a little deeper to find out more.

# COMMAND ----------

#%run ./0_data_preprocessing_seoul_bike_project

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from datetime import timedelta 

import plotly.express as px
import plotly.figure_factory as ff

%matplotlib inline
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Ingestion

# COMMAND ----------

#read in the parquet file, which was created in a previous notebook
df = pd.read_parquet('seoul_bike_dataset_final')
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### General EDA
# MAGIC 
# MAGIC 1. Distribution of bike rentals by date
# MAGIC 2. Impact of Hour
# MAGIC 3. Distribution for the holidays and seasons
# MAGIC 4. Distributions of the other continuous features
# MAGIC 5. Correlations between continous features and outcome

# COMMAND ----------

# MAGIC %md
# MAGIC #### Daily count of bikes rented

# COMMAND ----------

df_by_date = df.groupby(['date'])['rented_bike_count'].agg(['sum', 'max']).reset_index()
df_by_date.loc[df_by_date['date'] =='2018-10-06']

# COMMAND ----------

fig = px.scatter(df_by_date, x='date', y= 'sum')
fig.show()

# COMMAND ----------

fig = px.scatter(df, x='hour', y= 'rented_bike_count', color = 'season')
fig.show()

# COMMAND ----------

sns.pairplot(df[['rented_bike_count', 'temperature','humidity','wind_speed','visability','dew_point','solar_radiation','rainfall']])

# COMMAND ----------

# MAGIC %md
# MAGIC Dewpoint and temp look to be strongly correlated. Temp is strongly related to the outcome variable as well.

# COMMAND ----------

sns.heatmap(df.corr())

# COMMAND ----------

sns.boxplot( 'hour', 'rented_bike_count', data = df)

# COMMAND ----------

# MAGIC %md
# MAGIC Looking at these results, it appears as though there are categories forming here, broadly speaking peak (hours 8 and 18) and off-peak (between 23 and 6). 

# COMMAND ----------

df_peak = df.copy()

df_peak.loc[(df['hour'].isin(['8'])), 'hour_cat']='Morning_Peak'
df_peak.loc[(df['hour'].isin(['17','18','19'])), 'hour_cat']='Evening_Peak'
df_peak.loc[(df['hour'].isin(['2','3','4','5','6'])), 'hour_cat']='Off-Peak'
df_peak.loc[(df['hour'].isin(['7','9','10','11','12','13','14','15','20','21','22','23','0','1'])), 'hour_cat']='Regular'

#df_peak[[((df['hour'] == '18') | (df['hour']=='8')), 'peak']]=1
#df_peak[(df.hour >22 & df.hour<7), 'peak']=0
df_peak

# COMMAND ----------

sns.boxplot( 'hour_cat', 'rented_bike_count', data = df_peak)


# COMMAND ----------

# MAGIC %md
# MAGIC These look better seperated now

# COMMAND ----------

sns.boxplot('rented_bike_count', 'holiday', data = df)

# COMMAND ----------

sns.boxplot('rented_bike_count', 'season', data = df)

# COMMAND ----------

#to do create panel plot by holidy to see if outliers are the culprit
sns.boxplot('rented_bike_count', 'weekend', data = df)

# COMMAND ----------

df.columns

# COMMAND ----------

#temp
fig = px.scatter(df, x='date', y= 'temperature')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Descriptions of data elements that aren't commonly described
# MAGIC Dewpoints - The temp to which air must be cooled to saturate it. Negative most freq in winter. Can never be higher than the temp. <br>
# MAGIC Solar Radiation - Sunlight. The electromagnetic radiation of the sun
# MAGIC Functioning Day - Don't know what this is, let's look at it.

# COMMAND ----------

#check to see if we have any values where the dew point is bove the temp
df_final.loc[df_final.dew_point > df_final.temperature]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Major takeaways:
# MAGIC - Dew point and temp are highly correlated, consider dropping dew point because it is less correlated with the outcome variable.
# MAGIC - Winter is much lower than the other seasons - could this be related to winter olympics though?
# MAGIC - Other seasons show higher rental values with summer being the highest.
# MAGIC - Holidays tend to be lower than non-holidays indicating these rentals are most likely used for business functions. 
# MAGIC - Weekday/weekend only slighly different, major differences in outliers
# MAGIC - There are spikes in usage (not a linear trend) that occur in the 7-9 hour and then again in the 17-19 hour.
