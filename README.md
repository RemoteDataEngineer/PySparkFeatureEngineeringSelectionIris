# PySparkFeatureEngineeringSelectionIris
Use Iris dataset from Sklearn for Feature Engineering of Continuous Values and Feature Selection

# Run this notebook - Engineering and Feature Selection of Iris Dataset from SKLearn to kick off:
## 0Mount Data -
Mounts containers for storing processed files

## 1Preprocess Data -
Reads iris dataset from sklearn libraries and preprocesses dataframes for Features, Targets and Features + Targets and saves dataframes to Parquet files in mounted containers

## 2ProfileFeaturesAndTarget -
Reads in Features + Targets from Parquet files in mounted containers and Peforms Pandas Profiling on the entire dataframe. Identify which columns are not highly correlated with target and each other, identifies duplicates and rows / columns that are missing

## 3EngineeringContinousFeatures
Reads in Features from Parquet files in mounted containers and scales the values using different methods

## 4FeatureSelection
Reads in Features + Targets from Parquet files in mounted containers and computes / plots importance of each feature

