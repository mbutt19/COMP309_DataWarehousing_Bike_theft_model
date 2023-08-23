import pandas as pd
import os    
import seaborn as sns
import matplotlib.pyplot as plt  
import numpy as np   
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from imblearn.over_sampling import SMOTEN, ADASYN
import time
import datetime

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        Data exploration
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    Load and describe data columns, provide descriptions & types, ranges and values of elements 
# ---------------------------------------------------------------------------------------------
path = "./"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
df_raw = pd.read_csv(fullpath,sep=',',low_memory=False)
pd.options.mode.chained_assignment = None

# Data Assesment
print(df_raw)
print(df_raw.columns.values)
print(df_raw.shape)
print(df_raw.info()) 
# _____________________________________________________________________________________________
#    Statistical assessments including means, averages, and correlations.
# ---------------------------------------------------------------------------------------------
print(df_raw.describe())
print(df_raw.groupby("Status").count())
print(df_raw.groupby("Premises_Type").count())
print(df_raw.groupby("Status").mean())
print(df_raw.groupby("Premises_Type").mean())
print(df_raw.corr(method='pearson'))
# _____________________________________________________________________________________________
#    Missing data evaluations 
# ---------------------------------------------------------------------------------------------
print(len(df_raw)-df_raw.count())
# _____________________________________________________________________________________________
#    Graphs and visualizations 
# ---------------------------------------------------------------------------------------------
continuous_vars = df_raw[['Occurrence_DayOfYear', 'Report_DayOfYear', 'Report_Hour', 'Occurrence_Hour', 'Cost_of_Bike']]
#df_raw["Status"].value_counts().plot.pie(autopct='%.2f')
#sns.distplot(df_raw["Cost_of_Bike"], rug=True, hist=False, vertical = True)
#sns.lineplot(data=df_raw[['Occurrence_Hour', 'Cost_of_Bike']],x=df_raw['Occurrence_Hour'],y=df_raw['Cost_of_Bike'])
#sns.lineplot(data=df_raw[['Occurrence_DayOfYear', 'Cost_of_Bike']],x=df_raw['Occurrence_DayOfYear'],y=df_raw['Cost_of_Bike'])
#sns.lineplot(data=df_raw[['Occurrence_DayOfYear', 'Cost_of_Bike']],x=df_raw['Occurrence_DayOfYear'],y=df_raw['Cost_of_Bike'], hue=(df_raw["Status"]))
#sns.lineplot(data=df_raw[['Occurrence_Hour', 'Cost_of_Bike']],x=df_raw['Occurrence_Hour'],y=df_raw['Cost_of_Bike'])
#sns.lineplot(data=df_raw[['Occurrence_DayOfYear', 'Cost_of_Bike']],x=df_raw['Occurrence_DayOfYear'],y=df_raw['Cost_of_Bike'], hue=(df_raw["Premises_Type"]))
#sns.heatmap(continuous_vars,annot=True, cmap='coolwarm', linewidth=0.5)
#sns.pairplot(df_raw, hue="Premise_Type")
#sns.pairplot(df_raw, hue="Status")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                           Data modelling
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    Data transformations – handling missing data/categorical data/normalization/standardizations
# ---------------------------------------------------------------------------------------------
# Calculate Days between reporting and occurrence
df_raw['Days_Between'] = [(df_raw["Report_DayOfYear"][i] - df_raw["Occurrence_DayOfYear"][i]) 
                                  if (df_raw["Report_DayOfYear"][i] >= df_raw["Occurrence_DayOfYear"][i]) 
                                  else (df_raw["Report_DayOfYear"][i] - df_raw["Occurrence_DayOfYear"][i]) + 365 
                                         for i in range(len(df_raw["Occurrence_DayOfYear"]))]
# Calculate Hours between reporting and occurrence from report and occurrence timestamp
df_raw["Total_Hours_Between"] = df_raw["Report_Hour"]
for i in range(len(df_raw["Report_Date"])):
    df_raw["Total_Hours_Between"][i] = (pd.to_datetime(df_raw["Report_Date"][i]) - pd.to_datetime(df_raw["Occurrence_Date"][i])).total_seconds()/3600

# Drop rows where Status is UNKNOWN from table and across columns
df_raw = df_raw[df_raw.Status != "UNKNOWN"]
# Transforming 'Status' into binary [STOLEN] = 1, [RECOVERED] = 0 and add new column named 
df_raw['Status'] = [1 if b=='STOLEN' else 0 for b in df_raw.Status]

# Drop unneeded columns
df_selected = df_raw[["Premises_Type",'Bike_Type','Cost_of_Bike',"Days_Between","Total_Hours_Between"]]
df_target = df_raw["Status"]

# Fill out missing values
print(f"+++++ Before FillNA: +++++\n{len(df_selected)-df_selected.count()}")
df_selected.fillna(0,inplace=True)
print(f"+++++ After FillNA: +++++\n{len(df_selected)-df_selected.count()}")

# Categorical Handling
print(df_selected.info())
df_selected = pd.get_dummies(df_selected, columns=["Premises_Type", 'Bike_Type'], dummy_na=True)
print(df_selected.info())

# Standardize
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_selected)
scaled_df = pd.DataFrame(scaled_df, columns=df_selected.columns.values)
print(scaled_df.info())
print(scaled_df.describe())
# _____________________________________________________________________________________________
#    Feature selection 
# ---------------------------------------------------------------------------------------------
target=["Status"]
predictors=["Premises_Type", "Days_Between", 'Bike_Type', 'Cost_of_Bike',"Total_Hours_Between"]

# _____________________________________________________________________________________________
#    Train, test data splitting 
# ---------------------------------------------------------------------------------------------

# _____________________________________________________________________________________________
#    Managing imbalanced classes if needed. 
# ---------------------------------------------------------------------------------------------

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        Predictive model building
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     Use logistic regression and decision trees as a minimum– use scikit learn.
# ---------------------------------------------------------------------------------------------

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Model scoring and evaluation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     Present results as scores, confusion matrices, and ROC - use sci-kit learn
# ---------------------------------------------------------------------------------------------

# _____________________________________________________________________________________________
#     Select and recommend the best performing model
# ---------------------------------------------------------------------------------------------

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Deploying the model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     Using a flask framework arrange to turn your selected machine-learning model into an API.
# ---------------------------------------------------------------------------------------------

# _____________________________________________________________________________________________
#     Using the pickle module, arrange for Serialization & Deserialization of your model.
# ---------------------------------------------------------------------------------------------

# _____________________________________________________________________________________________
#     Build a client to test your model API service. Use the test data, which was not previously 
#     used to train the module. You can use simple Jinja HTML templates with or without Java , 
#     script REACT or any other technology but at minimum use POSTMAN Client API.
# ---------------------------------------------------------------------------------------------

