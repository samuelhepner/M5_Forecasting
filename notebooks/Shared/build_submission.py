# Databricks notebook source
# DBTITLE 1,Import libraries needed
import pandas as pd
import numpy as np

# COMMAND ----------

# DBTITLE 1,Get file paths
display(dbutils.fs.ls("dbfs:/FileStore/tables/"))

# COMMAND ----------

calendar = pd.read_csv("/dbfs/FileStore/tables/calendar.csv")
encoded = spark.read.csv("dbfs:/FileStore/tables/sales_train_validation_encoded.csv", header=True)
sample_submission = pd.read_csv("/dbfs/FileStore/tables/sample_submission.csv")
sell_prices = pd.read_csv("/dbfs/FileStore/tables/sell_prices.csv")
sales_train_valid = pd.read_csv("/dbfs/FileStore/tables/sales_train_validation.csv")

# COMMAND ----------

display(encoded)

# COMMAND ----------

sample_submission.head()

# COMMAND ----------

def melt_submission(data):
  """
  For melting the evaluation/validation data.
  """
  melted_data = data.melt(id_vars=['id'],
                          var_name='Date',
                          value_name='Predictions')
  return melted_data

# COMMAND ----------

def melt_sub(data):
  # Split into validation and evaluation since they cover different days
  validation = sample_submission[sample_submission['id'].str.contains("validation", case=True)]
  evaluation = sample_submission[sample_submission['id'].str.contains("evaluation", case=True)]
  
  # Verify shape of validation and evaluation
  print('# of rows in validation', validation.shape)
  print('# of rows in evaluation', evaluation.shape)
  print('# of rows in sample_submission', sample_submission.shape)
  assert(validation.shape[0] + evaluation.shape[0]), sample_submission.shape[0]
      
  # Get new column names
  validation.columns = ['id'] + ["d_" + str(x) for x in range(1914, 1942)]

  evaluation.columns = ['id'] + ["d_" + str(x) for x in range(1942, 1970)]
  
  # Melt validation and evaluation
  melted_val = melt_submission(validation)
  melted_eval = melt_submission(evaluation)
  
  # Concatenate melted_val and melted_eval back together
  sample_melt = pd.concat([melted_val, melted_eval])
  
  return sample_melt

# COMMAND ----------

melted_sample = melt_sub(sample_submission)
melted_sample.head()

# COMMAND ----------

def complete_sample(data, cal):
  # TO-DO: Add a docstring   
  
  # Make item_id, dept_id, cat_id, store_id, and state_id from id column
  # Adding some columns to match the encoded 
  
  # TO-DO: Use the .split functionality
  # https://www.tutorialspoint.com/python/string_split.htm
  # Does this work for all records?
  data['item_id'] = data['id'][0:1][0][0:13]
  data['dept_id'] = data['id'][0:1][0][0:9]
  data['cat_id'] = data['id'][0:1][0][0:7]
  data['store_id'] = data['id'][0:1][0][14:18]
  data['state_id'] = data['id'][0:1][0][14:16]
  
  # Get w_day and weekday column by joining with calendar table
  cal = cal.rename(columns = {'d': 'Date'})
  complete_sample_melted = melted_sample.merge(cal[['Date', 'wday', 'weekday']], how='left', on='Date')
  
  # Reorder columns
  cols = ['id','item_id','dept_id','cat_id','store_id','state_id','Date','weekday','wday','Predictions']
  complete_sample = complete_sample_melted[cols]
  
  # Rename Predictions column to Prediction
  complete_sample.rename(columns={"Predictions": "Prediction"})
  
  #  Write to csv   
#      complete_sample.to_csv('/dbfs/FileStore/tables/sample_melted.csv', index=False)
  
  return complete_sample  

# COMMAND ----------

sample = complete_sample(melted_sample, calendar)
sample.head()

# COMMAND ----------

# check to make sure it's saved in dbfs
display(dbutils.fs.ls("dbfs:/FileStore/tables/"))

# COMMAND ----------

