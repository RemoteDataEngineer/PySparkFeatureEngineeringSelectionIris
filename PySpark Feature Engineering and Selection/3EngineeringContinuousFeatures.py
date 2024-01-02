# Databricks notebook source
# MAGIC %md
# MAGIC ##Import libaries

# COMMAND ----------

print("0. Import libraries")
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in Datframes

# COMMAND ----------

print("1. Read in features of iris")
iris_dataset_features               = spark.read.parquet("/mnt/asdl2linearrg/iris/raw/iris_dataset_features.parquet")
display(iris_dataset_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Transforms features by scaling each feature to a given range

# COMMAND ----------

print("2. transform features 1-4 to have a range of 0")

# COMMAND ----------

features_array = np.array(iris_dataset_features.select("*").collect())

#print(features_array)

print("create scaler method")
scaler = MinMaxScaler(feature_range=(0,1))

print("fit and transform the data")
scaled_data = scaler.fit_transform(features_array)

feature_columns = ["feature1", "feature2", "feature3", "feature4"]

# Create PySpark DataFrame from NumPy array
df_MinMaxScaled_data = spark.createDataFrame(scaled_data.tolist(), feature_columns)

df_MinMaxScaled_data.show()

print("Write df_MinMaxScaled_data dataframes to parquet")
df_MinMaxScaled_data.write.mode("overwrite").parquet("/mnt/asdl2linearrg/iris/exCatAndContVariables/MinMax_iris_features.parquet")

# COMMAND ----------

print("3. transform features 1-4 to have a mean of 0 and std of 1")

# COMMAND ----------

print("create scaler method") 
scaler = StandardScaler()

print("fit and transform the data")
scaled_data = scaler.fit_transform(features_array )

print("mean of each feature is nearly 0:")
print(scaled_data.mean(axis=0))

print("standard deviation of each feature is 1:")
print(scaled_data.std(axis=0))

# Create PySpark DataFrame from NumPy array
df_StandardScaled_data = spark.createDataFrame(scaled_data.tolist(), feature_columns)

df_StandardScaled_data.show()

print("Write df_StandardScaled_data dataframes to parquet")
df_StandardScaled_data.write.mode("overwrite").parquet("/mnt/asdl2linearrg/iris/exCatAndContVariables/StandardScaled_iris_features.parquet")

