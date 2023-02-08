# Databricks notebook source
# MAGIC %md
# MAGIC ## Background/context
# MAGIC This dataset is in csv form and as a result has a bunch of fun challenges we need to dive into fixing so we can use it in any of our work.

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from datetime import timedelta 

%matplotlib inline
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data ingestion and pre-cleaning

# COMMAND ----------

#read in the table, which was loaded into databricks perviously
df = spark.read.table('seoul_bike_rentals_csv').toPandas()
# rename the columns so they are eaasier to work with
df.rename(columns = {"Date" : "date" 
                     , "Rented_Bike_Count" : "rented_bike_count"
                     , "Hour" : "hour"
                     , "Temperature_°C_" : "temperature"
                     , "Humidity_%_" : "humidity"
                     , "Wind_speed__m/s_" : "wind_speed"
                     , "Visibility__10m_" : "visability"
                     , "Dew_point_temperature_°C_" : "dew_point"
                     , "Solar_Radiation__MJ/m2_" : "solar_radiation"
                     , "Rainfall_mm_" : "rainfall"
                     , "Snowfall__cm_" : "snowfall"
                     , "Seasons" : "season"
                     , "Holiday" : "holiday"
                     , "Functioning_Day" : "functioning_day"                    
                    }
         , inplace = True)

#convert variables to the proper type starting with numeric
df[["rented_bike_count",  "temperature", "humidity", "wind_speed", "visability", "dew_point", "solar_radiation", "rainfall", "snowfall"]] = df[["rented_bike_count", "temperature", "humidity", "wind_speed", "visability", "dew_point", "solar_radiation", "rainfall", "snowfall"]].apply(pd.to_numeric)
#next convert categorical variables to string type
df[["hour", "season", "holiday", "functioning_day"]] = df[["hour", "season", "holiday", "functioning_day"]].astype('string')
#finally datetime
df["date"] = pd.to_datetime(df["date"])
df.head()

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Cleaning and variable creation

# COMMAND ----------

#first create a datetime variable for the date + hour
df['date_time'] = pd.to_datetime(df['date'].astype(str) + '' +df['hour'].astype(str) + ':00:00', format='%Y-%m-%d%H:%M:%S')

# COMMAND ----------

df_final = df.copy()
df_final.describe()
df_final.info()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Final output
# MAGIC Output the file for later ingestion

# COMMAND ----------

df_final.to_parquet("seoul_bike_dataset_final")
