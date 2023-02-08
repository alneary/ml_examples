# Databricks notebook source
--reminder of partitions

SELECT xxx LAG(elig_end_date) OVER (PARTITION BY person_id, sponsor_name ORDER BY date) as prev_date,


SELECT xxx SUM(break_in_elig) OVER (PARTITION BY xxxx) <=1

# COMMAND ----------

g = sns.FacetGrid(df, col = 'colname', row='rowname')
g.map(sns.scatterplot, "x", "y")
