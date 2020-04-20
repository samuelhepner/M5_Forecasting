# Databricks notebook source
# MAGIC %md <h1>Generate Submissions</h1>

# COMMAND ----------

# Spark Functions
from pyspark.sql import functions as F

# Python Functions
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Get File Paths
display(dbutils.fs.ls("dbfs:/FileStore/tables/"))

# COMMAND ----------

# DBTITLE 1,Create Baseline Submission using mean of Weekday by Item
# Load melted_submission file
sample_melted = pd.read_csv("/dbfs/FileStore/tables/sample_melted.csv")
sample_submission = pd.read_csv("/dbfs/FileStore/tables/sample_submission.csv")

# COMMAND ----------

train_calendar = spark.read.csv("dbfs:/FileStore/tables/sales_train_validation_calendar.csv", header=True) \
                      .select(['id','item_id','store_id','wday','Target'])
# display(train_calendar)

# COMMAND ----------

# DBTITLE 1,Create DataFrame with Target mean by Item, wday
item_wday_means_df = train_calendar.groupby(['id', 'item_id', 'store_id', 'wday']) \
                                   .agg(F.mean(F.col('Target')).alias('Target_mean')) \
                                   .toPandas()

# COMMAND ----------

item_wday_means_df[(item_wday_means_df['id'] == 'FOODS_3_090_WI_3_validation')]

# COMMAND ----------

item_wday_means_df.describe(percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99])

# COMMAND ----------

# DBTITLE 1,Replace Predictions in Sample Submission DataFrame with the Target_mean variable
print('Mean_df columns', item_wday_means_df.columns)
print('sample_melted columns', sample_melted.columns)

# COMMAND ----------

"""
sample_melted = data_1
item_wday_means_df = data_2
"""
def replace_preds(data_1, data_2):
  # Data 1 has wday column as int and data_2 has it as string objects
  # https://stackoverflow.com/questions/50649853/trying-to-merge-2-dataframes-but-get-valueerror
  data_2['wday'] = data_2['wday'].astype(int)
  
  # Merging datasets
  # https://www.geeksforgeeks.org/python-pandas-merging-joining-and-concatenating/
  preds = pd.merge(data_1, data_2.drop('id', axis = 1), on=['item_id','store_id','wday'])
  
  # Make sure there are no NaNs/nulls
  num_nulls = preds['Target_mean'].isnull().sum()
  assert num_nulls == 0
  
  # Copy values from Target_mean column to Predictions and get rid of Target_mean column
  preds['Predictions'] = preds['Target_mean']
  new_preds = preds[['id', 'item_id','dept_id','cat_id','store_id','state_id','Date','weekday','wday','Predictions']]
  
  if (new_preds.shape[0] != data_1.shape[0]):
    print("You're missing rows.")
  
  return new_preds

# COMMAND ----------

preds = replace_preds(sample_melted, item_wday_means_df)
preds.head()

# COMMAND ----------

# DBTITLE 1,Pivot the melted Sample Submission DataFrame so we can submit on Kaggle!
# use the .pivot function
# column names are F_1, F_2, ..., etc.
# put into a new "functions" notebook
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot.html
preds_pivot = preds.pivot(index='id',columns='Date', values='Predictions')
preds_pivot.head()

# COMMAND ----------

# preds_pivot.columns = preds_pivot.columns.droplevel(0)
# preds_pivot.columns.name = None
preds_pivot = preds_pivot.reset_index()
preds_pivot.head()

# COMMAND ----------

print(preds_pivot.index.name)

# COMMAND ----------

preds_pivot.columns.name = None
preds_pivot.head()

# COMMAND ----------

def back_to_f(data):
  # Split into validation and evaluation since they cover different days
  validation = data[data['id'].str.contains("validation", case=True)]
  evaluation = data[data['id'].str.contains("evaluation", case=True)]
  
  val_cols = ['id'] + ["d_" + str(x) for x in range(1914, 1942)]
  eval_cols = ['id'] + ["d_" + str(x) for x in range(1942, 1970)]
      
  valid = validation[val_cols]
  eval = evaluation[eval_cols]
  
  # Get new column names
  valid.columns = ['id'] + ["F" + str(x) for x in range(1, 29)]
  eval.columns = ['id'] + ["F" + str(x) for x in range(1, 29)]
  
  # Concatenate melted_val and melted_eval back together
  submission = pd.concat([valid, eval])
  
  return submission

# COMMAND ----------

def make_submission(df, sample):
  """
  Does your table look almost exactly like the sample submission file but the id column is out of order?
  Then you've come to the right place.
  """
  # Copy sample file and set index to 'id' column
  x = sample.copy()
  x.set_index('id', inplace=True)
  
  # copy the file that you want to turn into a submission and set the index to 'id'
  y = df.copy()
  y.set_index('id', inplace=True)
  
  # get your id column to be sorted like the sample submission id column
  submission = y.loc[x.index]
  submission = submission.reset_index()
  
  return submission

# COMMAND ----------

df = back_to_f(preds_pivot)
df.head()

# COMMAND ----------

new_submission = make_submission(df, sample_submission)
display(new_submission)

# COMMAND ----------

# DBTITLE 1,Make 5 Submissions with different combinations of features
# id, wday
# id, wday, month
# id, state, wday
# id, state, wday, month

# Note score of each submission!

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Plot the time series for 5 different unique ids over time for each of your 5 submissions above
# you will need to combine the sales_train_validation_melted with your sample submission to do this!
# color the training data with one color and color the test data with another color

# using bokeh() visually inspect the time series
# do these look like reasonable predictions? Are they way off?

# COMMAND ----------

# DBTITLE 0,Load in sales_train_validation_melted
sales_train_validation_melted = spark.read.csv("dbfs:/FileStore/tables/sales_train_validation_melted.csv", header=True)
sales_h1_valid = sales_train_validation_melted[sales_train_validation_melted['id'] == 'HOBBIES_1_001_CA_1_validation'].toPandas()
sales_h1_valid.tail()

# COMMAND ----------

# DBTITLE 1,Match the columns that the submission has
sales_id_Date_Target = sales_h1_valid[['id', 'Date', 'Target']]
sales_id_Date_Target.head()

# COMMAND ----------

# DBTITLE 1,Prepping new submission for merge with sales_train_validation_calendar
new_submission.head()

# COMMAND ----------

# DBTITLE 1,Melts data and doesn't matter if validation/evaluation
def melt_id(data, id):
  """
  Melts data when in validation
  """
  id_preds = data[data['id'] == id]
  if id.find("validation") >= 0:
    new_column_names = ['id'] + ['d_' + str(x) for x in range(1914, 1942)]
    value_cols = ['d_' + str(x) for x in range(1914, 1942)]
  else:
    new_column_names = ['id'] + ['d_' + str(x) for x in range(1942, 1970)]
    value_cols = ['d_' + str(x) for x in range(1942, 1970)]
  
  id_preds.columns = new_column_names
  id_preds_melted = pd.melt(id_preds, id_vars=['id'], value_vars=value_cols,
                                 var_name = 'Date', value_name = 'Preds')
  return id_preds_melted

# COMMAND ----------

id_preds_melted = melt_id(new_submission, 'HOBBIES_1_001_CA_1_validation')
id_preds_melted.head()

# COMMAND ----------

def combine_data(data1, data2):
  """
  Concats the data and combines the Preds and Target columns into new column called Predictions
  """
  history = pd.concat([data1, data2], sort=False)
  
  # Get rid of NaNs
  history['Preds'].fillna(0, inplace=True)
  history['Target'].fillna(0, inplace=True)
  
  # One column to rule them all
  history['Predictions'] = history['Target'].astype(int) + history['Preds']
  
  # Get rid of the other columns
  id_date_predictions = history[['id', 'Date', 'Predictions']]
  
  return id_date_predictions

# COMMAND ----------

id_date_predictions = combine_data(sales_id_Date_Target, id_preds_melted)
id_date_predictions.head()

# COMMAND ----------

# DBTITLE 1,Graphing but I think I need to change the values in the Date column to be numbers.
from bokeh.plotting import figure, output_file, show

output_file('id_predictions.html')

# COMMAND ----------

p = figure()
p.circle(x='Date', y='Predictions',
        source=id_date_predictions,
        size=10, color='green')

# COMMAND ----------

p.title.text = 'Sales for Hobbies_1_001_CA_1_validation'
p.xaxis.axis_label = 'Dates'
p.yaxis.axis_label = 'Sales'

# COMMAND ----------

show(p)