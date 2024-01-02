# Databricks notebook source
# MAGIC %md
# MAGIC ##Import libraries

# COMMAND ----------

print("0. Import libraries")

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
  
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Iris Data From SKlearn

# COMMAND ----------

print( "1. Load Iris Data From SKlearn")

# Load iris data
print("iris dataset from sklearn")
iris_dataset = load_iris()
display(type(iris_dataset))


# COMMAND ----------

# MAGIC %md
# MAGIC #Dataframes for target, features and target + features 

# COMMAND ----------

print( "2. Seperate columns in Iris from features (x) and target (y)")

# COMMAND ----------

# Create features and target
print("features of iris datasets")
X = iris_dataset.data
display(X)
display(type(X))
display(X.shape)

print("targets of iris datasets")
y = iris_dataset.target
display(y)
display(type(y))
display(y.shape)

# COMMAND ----------

print("create a schema for the dataframe featuring 2 numpy arrays X andy ")
schema = StructType([
    StructField("feature1", FloatType(), True),
    StructField("feature2", FloatType(), True),
    StructField("feature3", FloatType(), True),
    StructField("feature4", FloatType(), True),
    StructField("target", IntegerType(), True)
])

# Convert NumPy arrays of Features and Targets to Python lists and then create a PySpark DataFrame
print("zip features and target into one list")
list_data = list(zip(X[:, 0].tolist(), X[:, 1].tolist(), X[:, 2].tolist(), X[:, 3].tolist(), y.tolist()))
type(list_data)

print("create df_iris_data ")
df_iris_data = spark.createDataFrame(list_data, schema=schema)
print(type(df_iris_data ))
print(df_iris_data.count() )

print("create a schema for the features of the iris dataset")
schema = StructType([
    StructField("feature1", FloatType(), True),
    StructField("feature2", FloatType(), True),
    StructField("feature3", FloatType(), True),
    StructField("feature4", FloatType(), True)
])


print("create df_iris_features ")
df_iris_features = spark.createDataFrame(X.tolist(), schema=schema)
print(type(df_iris_features ))
print(df_iris_features.count() )

print("create a schema for the target of the iris dataset")
schema = StructType([
    StructField("target", IntegerType(), True)
])


print("Convert target values to Python integers")
y_python_integers = [int(value) for value in y]

print("Create df_iris_target DataFrame")
df_iris_target = spark.createDataFrame([(value,) for value in y_python_integers], schema=schema)



print(type(df_iris_target))
print(df_iris_target.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write dataframes

# COMMAND ----------

print("Write new iris dataframes to parquet")
df_iris_data.write.mode("overwrite").parquet("/mnt/asdl2linearrg/iris/raw/iris_dataset.parquet")
df_iris_features.write.mode("overwrite").parquet("/mnt/asdl2linearrg/iris/raw/iris_dataset_features.parquet")
df_iris_target.write.mode("overwrite").parquet("/mnt/asdl2linearrg/iris/raw/iris_dataset_target.parquet")
