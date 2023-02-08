# Databricks notebook source


# COMMAND ----------

import plotly.express as px
import plotly.figure_factory as ff

# COMMAND ----------

fig = px.scatter(df_by_date, x='date', y= 'rented_bike_count', color = 'season', symbol = 'holiday')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlations

# COMMAND ----------

# MAGIC %md
# MAGIC ### Countplots

# COMMAND ----------

sns.countplot(x='species', hue='sex', data=penguins)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Pairplots

# COMMAND ----------

sns.pairplot(penguins[['species', 'sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']], hue='species')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Regression Plots

# COMMAND ----------

sns.regplot('flipper_length_mm', 'body_mass_g', data=penguins, line_kws={'color': 'orange'})

