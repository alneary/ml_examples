# Databricks notebook source
# Use secrets DBUtil to get Snowflake credentials.
#user = dbutils.secrets.get("data-warehouse", "svc_demo")
#password = dbutils.secrets.get("data-warehouse", "demo123!")

# snowflake connection options
options = {
  "sfUrl": "ha31264.us-east-2.aws",
  "sfUser": "svc_demo",
  "sfPassword": "demo123!",
  "sfDatabase": "DEMO_DB",
  "sfSchema": "Exercise",
  "sfWarehouse": "EXERCISE_WH"
}

# COMMAND ----------

# MAGIC %md
# MAGIC Write to snowflake

# COMMAND ----------

# Generate a simple dataset containing five values and write the dataset to Snowflake.
spark.range(5).write 
  .format("snowflake") 
  .options(**options) 
  .option("dbtable", "<snowflake-database>") 
  .save()

# COMMAND ----------

# MAGIC %md
# MAGIC Read/Display Dataset from snowflake

# COMMAND ----------

# Read the data written by the previous cell back.
df = (spark.read
    .format("snowflake") 
    .options(**options) 
    .option("query", "SELECT * from CI_ORDER") 
    .load()
     )
display(df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Query Data in Snowflake

# COMMAND ----------

df = spark.read 
  .format("snowflake") 
  .options(**options) 
  .option("query",  "select 1 as my_num union all select 2 as my_num") 
  .load()

df.show()

