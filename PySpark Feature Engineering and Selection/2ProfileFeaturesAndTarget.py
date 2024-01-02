# Databricks notebook source
# MAGIC %md
# MAGIC ##Import libaries

# COMMAND ----------

print("00. Installation")
%pip install --upgrade ydata_profiling
%pip install typing_extensions==4.7.1 --upgrade

dbutils.library.restartPython()



# COMMAND ----------

print("0. Import libraries")
from ydata_profiling import ProfileReport
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in Datframes

# COMMAND ----------

print("1. Read in features and targets of iris")
iris_dataset_df               = spark.read.parquet("/mnt/asdl2linearrg/iris/raw/iris_dataset.parquet")
display(iris_dataset_df)

# COMMAND ----------

print("2. Generate a profiling report")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Print results of Pandas profiler

# COMMAND ----------

print("Results. Feature 2 is not as highly correlated witht the Target as Feature 1, Feature 3, Feature 4")
print("........ Number of variables	5")
print("........ Number of observations	150")
print("........ Missing cells	0")
print("........ Missing cells (%)	0.0%")
print("........ Duplicate rows	1")
print("........ Duplicate rows (%)	0.7%")
print("........ profile = ProfileReport(iris_dataset_df)")
print("........ profile.to_notebook_iframe()")
