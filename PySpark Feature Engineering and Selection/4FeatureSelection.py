# Databricks notebook source
# MAGIC %md
# MAGIC ##Import Libraries

# COMMAND ----------

print("0. Import libraries")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in Dataframes

# COMMAND ----------

print("1. Read in Standard Scaled features of iris")
iris_StandardScaled_df               = spark.read.parquet("/mnt/asdl2linearrg/iris/exCatAndContVariables/StandardScaled_iris_features.parquet")

iris_Features_df                     = spark.read.parquet("/mnt/asdl2linearrg/iris/raw/iris_dataset_target.parquet")
display(iris_StandardScaled_df )
display(iris_Features_df       )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance

# COMMAND ----------

print("2. Feature importance")

print("array named X for Standard Scaled features")
X = np.array(iris_StandardScaled_df.select("*").collect())

print("array named y for target")
y = np.array(iris_Features_df.select("*").collect())


# COMMAND ----------

print("Building ExtraTreesClassifier")
extra_tree_forest = ExtraTreesClassifier(n_estimators = 5,
                                        criterion ='entropy', max_features = 2)
  
print("Training the model with features and target arrays")
extra_tree_forest.fit(X, y)
  
print("Computing the importance of each feature")
feature_importance = extra_tree_forest.feature_importances_
  
print("Normalizing the individual importances")
feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                        extra_tree_forest.estimators_],
                                        axis = 0)

# COMMAND ----------

print("Plot bar graph to compare models")

print("create list of feature columns")
feature_columns = ["feature1", "feature2", "feature3", "feature4"]

# Plotting a Bar Graph to compare the models
plt.bar(feature_columns, feature_importance_normalized)
plt.xlabel('Feature Labels')
plt.ylabel('Feature Importances')
plt.title('Comparison of different Feature Importances')
plt.show()
