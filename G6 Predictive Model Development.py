# -*- coding: utf-8 -*-
"""
Created Fall 2021

Course: COMP309

@authors - Group 6:
    Butt, Mustafa
    Galang, Romnick
    McKenzie, Shaniquo
    Sayed Rahim, Fawad    
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json

'''

╔══════════════════════════════════════════════════════════════════════╗
✧･ﾟ: *✧･ﾟ:* 　⋆﹥━━━━━━━━━ 1. Data Exploration ━━━━━━━━━﹤⋆　 *:･ﾟ✧*:･ﾟ✧
╚══════════════════════════════════════════════════════════════════════╝

'''

'''
─────────────────────────────── ❝ 1.1 ❞ ──────────────────────────────
  Load and describe data elements (columns), provide descriptions & 
  types, ranges and values of elements as appropriate - use pandas, 
  numpy and any other python packages.
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
'''
# Load the dataset in a dataframe object 
path = r"."
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path, filename)

df = pd.read_csv(fullpath, low_memory=False)


# method for data exploration
def data_exploration():
    # prints the dataset
    # print(df)

    # Explore the data check the column values
    # print the column names
    print(df.columns.values)

    # prints the specified number of rows in the dataset
    print(df.head(3))

    # give option to show only a set number of columns in the format below
    # firstColumn ... lastColumn
    pd.set_option('display.max_columns', 5)

    # prints the first five rows in the dataset with the number of columns specified, (Default = 5)
    print(df.head())

    # prints the num of non-null data entries, data number and column type
    print(df.info())

    # prints the number of rows and columns in the dataset
    print(df.shape)

    # prints the data type of each cloumn in the dataset
    print(df.dtypes)

    # prints the unique values in the specified columns
    print(df['Bike_Type'].unique())
    print(df['Bike_Colour'].unique())
    print(df['Occurrence_Year'].unique())
    print(df['Primary_Offence'].unique())
    print(df['City'].unique())
    print(df['Division'].unique())

    # Counts of values by column groupings to check count densities across column's unique values
    print(df.groupby("Status").count())
    print(df.groupby("Primary_Offence").count())
    print(df.groupby("Division").count())
    print(df.groupby("City").count())
    print(df.groupby("Premises_Type").count())
    print(df.groupby("Bike_Colour").count())
    print(df.groupby("Bike_Type").count())
    print(df.groupby("Occurrence_Year").count())
    print(df.groupby("Report_Year").count())
    print(df.groupby("Occurrence_Month").count())
    print(df.groupby("Report_Month").count())
    print(df.groupby("Occurrence_DayOfWeek").count())
    print(df.groupby("Report_DayOfWeek").count())

    # prints the unique values and number of unique values in each columnof the dataset
    cols = df.columns.values
    for i in range(len(cols)):
        print(f'{i + 1} - {cols[i]} has {df[cols[i]].nunique()} values ')
        print(f'These are:\n\t {df[cols[i]].unique()} \n\n ')

    '''
    ────────────────────────────── ❝ 1.2 ❞ ──────────────────────────────
      Statistical assessments including means, averages, and correlations.
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    '''

    # prints the unique values in the specified column
    print("\nFrequency of occurance at certain hours\n", df['Occurrence_Hour'].value_counts())
    print("\nFrequency of occurance within a specified month\n", df['Occurrence_Month'].value_counts())
    print("\nFrequency of occurance for a specified week day\n", df['Occurrence_DayOfWeek'].value_counts())
    print("\nFrequency of occurance in a certain neighbourhood\n", df['NeighbourhoodName'].value_counts())
    print("\nStatus count for the bikes\n", df['Status'].value_counts())

    # Gives statistical descriptions (eg mean, min, max, standard deviation and interquartie range) of the numeric dataset
    print(df.describe())

    # prints the mean of each column according to the occurance year
    print(df.groupby('Occurrence_Year').mean())

    # prints the mean of each column according to the Status
    print(df.groupby('Status').mean())

    # prints the mean standard deviation of each column according to the status
    print(df.groupby('Status').mean().std())
    print(df.groupby('Occurrence_DayOfWeek').mean().std())
    print(df.groupby('Report_DayOfMonth').mean().std())

    # Group data by a column's unique values and get statistical assessments
    print("\n", df.groupby("Division").describe())
    print("\n", df.groupby("Bike_Type").describe())
    print("\n", df.groupby("Premises_Type").describe())
    print("\n", df.groupby("Occurrence_Year").describe())
    print("\n", df.groupby("Report_Year").describe())
    print("\n", df.groupby("Occurrence_Month").describe())
    print("\n", df.groupby("Report_Month").describe())
    print("\n", df.groupby("Occurrence_DayOfWeek").describe())
    print("\n", df.groupby("Report_DayOfWeek").describe())

    # Check % of frequeency for top 100 unique values of total value distribution
    print(df['Bike_Colour'].value_counts().head(10).values.sum() / df['Bike_Colour'].value_counts().sum())
    print(df['Bike_Type'].value_counts().head(10).values.sum() / df['Bike_Type'].value_counts().sum())
    print(df['Bike_Make'].value_counts().head(10).values.sum() / df['Bike_Make'].value_counts().sum())
    print(df['Bike_Model'].value_counts().head(1000).values.sum() / df['Bike_Model'].value_counts().sum())
    print(df['Bike_Colour'].value_counts().head(10).values.sum() / df['Bike_Colour'].value_counts().sum())
    print(df['Division'].value_counts().head(10).values.sum() / df['Division'].value_counts().sum())
    print(df['NeighbourhoodName'].value_counts().head(10).values.sum() / df['NeighbourhoodName'].value_counts().sum())
    print(df['Report_DayOfYear'].value_counts().head(100).values.sum() / df['Report_DayOfYear'].value_counts().sum())
    print(df['Occurrence_DayOfYear'].value_counts().head(100).values.sum() / df[
        'Occurrence_DayOfYear'].value_counts().sum())

    # To find the correlation among
    # the columns using kendall method
    # Use corr() function to find the correlation among the columns in the dataframe using ‘pearson’ method.
    df.corr(method='pearson')

    '''
    ────────────────────────────── ❝ 1.3 ❞ ──────────────────────────────
       Missing data evaluations – use pandas, numpy, and any other 
       python packages.
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    '''

    # Print a summary of all missing values in all columns
    print(df.isna().sum())
    # prints the num of non-null data entries, data number and column type
    print(df.info())
    # prints the num of non-null data entries
    print(df.count())
    # prints total num of rows/data entries
    print(len(df))
    # check for null data entries/values by subtracting null values from the total row count
    print(len(df) - df.count())

    '''
    ────────────────────────────── ❝ 1.4 ❞ ──────────────────────────────
      Graphs and visualizations – use pandas, matplotlib, seaborn, numpy, 
      and any other python packages. You can also use power BI desktop.
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    '''

    # plot a pie chart for Status column values
    df["Status"].value_counts().plot.pie(autopct='%.2f', )

    # plots a histogram for the Status column
    plt.title('Histogram of Status')
    plt.xlabel('Status')
    plt.ylabel('Frequency')
    plt.hist(df['Status'])

    # Use seaborn library to generate different plots:
    plt.title('Reports per year')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    sns.distplot(df['Report_Year'])

    # Change the direction of the plot
    plt.title('Occurance per year')
    plt.xlabel('Frequency')
    sns.distplot(df['Occurrence_Year'], rug=True, hist=False, vertical=True)

    # change the color of the plot
    plt.title('Cost of bike')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    sns.distplot(df['Cost_of_Bike'], rug=True, hist=False, color='g')

    # plots histogram of report year
    df.Report_Year.hist()
    plt.title('Histogram of Report_Year')
    plt.xlabel('Report_Year')
    plt.ylabel('Frequency')

    # show the status of the bikes on a monthly basis with regards to occurance month
    pd.crosstab(df.Occurrence_Month, df.Status).plot(kind='bar')
    plt.title('Status Frequency for each Occurrence_Month')
    plt.xlabel('Occurrence_Month')
    plt.ylabel('Frequency of Status')

    # shows how many report made during a particular year for a specific day of the week
    pd.crosstab(df.Report_Year, df.Report_DayOfWeek).plot(kind='line')
    plt.title('Report_DayOfWeek Frequency for each Report_Year')
    plt.xlabel('Report_Year')
    plt.ylabel('Frequency of Report_DayOfWeek')

    # show report year frequency for each day of the week on a box plot
    pd.crosstab(df.Report_Year, df.Report_DayOfWeek).plot(kind='box')
    plt.title('Report_Year Frequency for each Report_DayOfWeek')
    plt.xlabel('Report_DayOfWeek')
    plt.ylabel('Frequency of Report_Year')

    # shows a line graph of bike types and their status, if its stolen, recovered 
    pd.crosstab(df.Bike_Type, df.Status).plot.barh()
    plt.title('Status Frequency for each  Bike_Type')
    plt.xlabel('Bike_Type')
    plt.ylabel('Frequency of Status')
    """
    # Check all correlations. Here it take longer time to execute 
    graphDf = sns.pairplot(df, hue="Status")
    graphDf.fig.suptitle("Corelation between all numeric columns in the dataset", y=1.05)
    
    # find corelation of a subset of two and three columns
    # only numeric datatypes can be used, but categoricals can be applied as hues
    x = df[['Report_Year','Occurrence_Year','Cost_of_Bike']]
    xHue = df[['Report_DayOfWeek','Occurrence_DayOfWeek','Cost_of_Bike', 'Status']]
    y = df[['Occurrence_DayOfYear','Cost_of_Bike']]
    yHue = df[['Occurrence_DayOfYear','Report_DayOfYear', 'Status']]
    
    # check the correlations 
    graphX = sns.pairplot(x)
    graphX.fig.suptitle("Corelation between Report_Year, Occurrence_Year and Cost_of_Bike", y=1.05)
    
    #Corelation between Report_DayOfYear, Occurrence_DayOfYearYear and Cost_of_Bike, by Status
    graphX2 = sns.pairplot(xHue, hue="Status")
    graphX2.fig.suptitle("Corelation between Report_DayOfYear, Occurrence_DayOfYearYear and Cost_of_Bike, by Status", y=1.05)
    
    #Corelation between Occurrence_DayOfYear and Cost_of_Bike
    graphY = sns.pairplot(y)
    graphY.fig.suptitle("Corelation between Occurrence_DayOfYear and Cost_of_Bike", y=1.05)
    
    #Corelation between Occurrence_DayOfYear and Report_DayOfYear, by Status
    graphY2 = sns.pairplot(yHue, hue='Status')
    graphY2.fig.suptitle("Corelation between Occurrence_DayOfYear and Report_DayOfYear, by Status", y=1.05)
    
    
    # Generate heatmaps
    #Heatmap of corelation between Occurrence_DayOfYear and Cost_of_Bike
    ax = plt.axes()
    hmap = sns.heatmap(y.corr(),annot=True, ax=ax, cmap='YlOrBr',)
    ax.set_title("Heatmap of corelation between Occurrence_DayOfYear and Cost_of_Bike", y=1.05)
    
    #Heatmap of corelation between Occurrence_DayOfYear and _DayOfYear
    ax = plt.axes()
    hmap = sns.heatmap(yHue.corr(),annot=True, ax=ax, cmap='YlOrBr')
    ax.set_title("Heatmap of corelation between Occurrence_DayOfYear and Report_DayOfYear", y=1.05)
    
    #Heatmap of corelation between Report_Year, Occurrence_Year and Cost_of_Bike
    ax = plt.axes()
    plt.figure(figsize=(10,9))
    sns.heatmap(x.corr(),annot=True, cmap='coolwarm', linewidth=0.5, ax=ax)
    ax.set_title("Heatmap of corelation between Report_Year, Occurrence_Year and Cost_of_Bike", y=1.05)
    
    
    ##line two variables
    plt.figure(figsize=(20,9))
    sns.lineplot(data=y,x='Occurrence_DayOfYear',y='Cost_of_Bike').set(title='Line plot of Occurrence_DayOfYear vs Cost_of_Bike before normalization')
"""
    ## line three variables
    sns.lineplot(data=df[['Occurrence_DayOfYear', 'Occurrence_Year', 'Cost_of_Bike']]).set(
        title='Line plot of Occurrence_DayOfYear, Occurrence_Year and Bike_Speed before normalization')


# calling the data exploration mmethod
# data_exploration()

'''

╔════════════════════════════════════════════════════════════════════╗
✧･ﾟ: *✧･ﾟ:* 　⋆﹥━━━━━━━━━ 2. Data Modelling ━━━━━━━━━﹤⋆　 *:･ﾟ✧*:･ﾟ✧
╚════════════════════════════════════════════════════════════════════╝

'''

'''
────────────────────────────── ❝ 2.1 ❞ ──────────────────────────────
  Data transformations – includes handling missing data, categorical 
  data management, data normalization, and standardizations as needed.
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
'''

# Transforming 'Status' into binary [STOLEN] = 1, [RECOVERED] = 0 and add new column named
# remove rows by filtering
df = df[df['Status'] != 'UNKNOWN']
print(df['Status'].value_counts())

df['BinaryStatus'] = [0 if s == 'RECOVERED' else 1 for s in df['Status']]
print(df['BinaryStatus'].value_counts())

# Calculate Hours between reporting and occurrence from Report_Date and Occurrence_Date & return as int64 column
df["Hours_Between"] = (pd.DatetimeIndex(df["Report_Date"]) - pd.DatetimeIndex(df["Occurrence_Date"])) * 24
df['Hours_Between'] = [(val / np.timedelta64(1, 'h')) for val in df["Hours_Between"]]
print(df.Hours_Between.describe)

# Transform Cost column to grouping of ranges
print(df.Cost_of_Bike.describe())
print(df.Cost_of_Bike.median(), df.Cost_of_Bike.mode())
print(df.Cost_of_Bike.value_counts().head(100), "\n\n", df.Cost_of_Bike.value_counts().tail(100))
print(df.Cost_of_Bike.value_counts().sort_index().head(100), "\n\n",
      df.Cost_of_Bike.value_counts().sort_index().tail(100))

print(len(df.Cost_of_Bike) - df.Cost_of_Bike.count())
df.Cost_of_Bike.fillna(df.Cost_of_Bike.mean(), inplace=True)
print(len(df.Cost_of_Bike) - df.Cost_of_Bike.count())


# Transformation method for Cost_of_Bike
def costGrouper(val):
    if 0 < val <= 300:
        return 300.
    elif 500 < val <= 1000:
        return 1000.
    elif 1000 < val <= 1500:
        return 1500.
    elif 1500 < val <= 2000:
        return 2000.
    elif 2000 < val <= 2500:
        return 2500.
    elif 2500 < val <= 3000:
        return 3000.
    elif 3000 and val <= 3500:
        return 3500.
    elif 3500 < val <= 4000:
        return 4000.
    elif 4000 < val <= 4500:
        return 4500.
    elif val > 5000:
        return 5000.
    else:
        return 0.


# Transform Cost_of_Bike to groupings by value
print(df["Cost_of_Bike"].value_counts())
df["Cost_of_Bike"] = df["Cost_of_Bike"].apply(costGrouper)
print(df["Cost_of_Bike"].value_counts())

# List of Features to focus on --> check significance then update the feature selection (select columns to use in analysis)
# Features that will definitely NOT be included
excluded = ['X', 'Y', 'OBJECTID', 'event_unique_id', 'Occurrence_Date', 'Report_Date', 'City',
            'Bike_Make', 'Bike_Model', 'Bike_Colour', 'Status', 'Longitude', 'Latitude', 'ObjectId2']

# Features that can potentially be used in first phase of feature selection 
### DO NOT REMOVE ! ! ! USE AS MASTER LIST
selectable = ['Primary_Offence', 'Occurrence_Year', 'Occurrence_Month', 'Occurrence_DayOfWeek', 'Occurrence_DayOfMonth',
              'Occurrence_DayOfYear', 'Occurrence_Hour', 'Report_Year', 'Report_Month', 'Report_DayOfWeek',
              'Report_DayOfMonth', 'Report_DayOfYear', 'Report_Hour', 'Division', 'NeighbourhoodName', 'Hood_ID',
              'Hours_Between',
              'Location_Type', 'Premises_Type', 'Bike_Type', 'Bike_Speed', 'Cost_of_Bike', 'BinaryStatus']

'''
#Best Scoring (67.3%)   => #Features=3
['Occurrence_DayOfWeek', 'Occurrence_DayOfYear', 'Report_DayOfWeek', 'Division',
            'NeighbourhoodName','Hood_ID', 'Location_Type', 'Premises_Type' , 
            'Bike_Type','BinaryStatus','Primary_Offence' ]
'''

# Features that will be used in first phase of feature selection
selected = ['Occurrence_DayOfWeek', 'Occurrence_DayOfYear', 'Report_DayOfWeek', 'Division',
            'NeighbourhoodName', 'Hood_ID', 'Location_Type', 'Premises_Type',
            'Bike_Type', 'BinaryStatus', 'Primary_Offence']

'''
Top 3 features:
 
         Feature  Support  Rank
2  Premises_Type     True     1
3      Bike_Type     True     1
7    Report_Hour     True     1

Top 5 features:
 
              Feature  Support  Rank
4            Division     True     1
7       Premises_Type     True     1
8           Bike_Type     True     1
14  Report_DayOfMonth     True     1
16        Report_Hour     True     1

Top 8 features:
 
                  Feature  Support  Rank
4                Division     True     1
7           Premises_Type     True     1
8               Bike_Type     True     1
10  Occurrence_DayOfMonth     True     1
12        Occurrence_Hour     True     1
14      Report_DayOfMonth     True     1
16            Report_Hour     True     1
18             Bike_Speed     True     1

Top 10 features:
 
                  Feature  Support  Rank
3        Report_DayOfWeek     True     1
4                Division     True     1
6           Location_Type     True     1
7           Premises_Type     True     1
8               Bike_Type     True     1
10  Occurrence_DayOfMonth     True     1
12        Occurrence_Hour     True     1
14      Report_DayOfMonth     True     1
16            Report_Hour     True     1
18             Bike_Speed     True     1

'''

# Collect column names of selected features
featuresToInclude = df[selected].columns.values

# assign columns to the dataframe
df_ = df[featuresToInclude]

'''Handle Missing values if any_______________________________________________________________________________________________________'''

# check for missing values
missingVal = None
df_label = ''
print(len(df_[featuresToInclude]) - df_[featuresToInclude].count())

# find missing columns in the features provided
if featuresToInclude.any():
    missingVal = pd.DataFrame(df_[featuresToInclude].isnull().sum())
else:
    missingVal = pd.DataFrame(df_.isnull().sum())

out_msg = "\nSum of Missing Values"
if df_label:
    out_msg += " for " + df_label

# print missing values
print(out_msg)
print("=" * len(out_msg))
print(missingVal[missingVal.iloc[:, 0] > 0])

'''df_ data column type seperation___________________________________________________________________________________________________'''
# check for categorical data or numeric

column_categoricals = []
column_dates = []
column_others = []
column_numerics = []
df_datatype_categories = pd.DataFrame()

# loop through and handle missing data based on the data type of the column
# also add the column type to their respective arrays

for col, col_type in df_.dtypes.iteritems():

    # check id data type is object and add to categorical array
    if col_type == 'object':
        # fillna method is used to replace missing values with frequently occuring values
        df_[col] = df_[col].fillna(df_.mode()[col])
        # print('object/categorical datatype: ', col_type)
        column_categoricals.append(col)
        df_datatype_categories = df_datatype_categories.append(
            {'Assigned Category': 'Object', 'Actual Datatype': col_type}, ignore_index=True)

    # if data is neither date time nor object, add it to the numeric array
    elif col_type != 'object' and not pd.api.types.is_datetime64_any_dtype(df_[col]):
        df_[col].fillna(df_[col].mean(), inplace=True)  # POTENTIAL
        # print('numerical datatype: ', col_type)
        column_numerics.append(col)
        df_datatype_categories = df_datatype_categories.append(
            {'Assigned Category': 'Number', 'Actual Datatype': col_type}, ignore_index=True)

    # when all else fails just use the default
    # note that the rows with missing data will be droped
    else:
        # fillna method is used to replace missing values with average
        df_.dropna(axis=0, how='any', inplace=True)
        # print('undefined datatype: ', col_type)
        column_others.append(col)
        df_datatype_categories = df_datatype_categories.append(
            {'Assigned Category': 'Undefined', 'Actual Datatype': col_type}, ignore_index=True)

print('Data Frame with the actual datatypes and categories assigned after sorting\n ', df_datatype_categories)
# print  column names based on data type category
print("\nCategorical Columns: ", column_categoricals)
print("\nNumeric Columns: ", column_numerics)
print("\nDate Columns: ", column_dates)
print("\nOther Columns: ", column_others)

# print update to show missing values were removed
df_.info()

'''Handle Categorical data _______________________________________________________________________________________________________'''

# assign numeric values to the categories
result = df_[column_categoricals].apply(lambda col: pd.Categorical(col).codes)

# join all columns in a new dataframe
df_ohe = pd.concat([result, df_[column_numerics]], axis=1, join="inner")
print(df_ohe.info())

'''
────────────────────────────── ❝ 2.2 ❞ ──────────────────────────────
   Feature selection – use pandas and sci-kit learn.
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
'''

'''Assign features _______________________________________________________________________________________________________'''

# Import your necessary dependencies
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

array = df_ohe.values
Y = array[:, df_ohe.columns.get_loc('BinaryStatus')]
X = [i for i in df_ohe if i != 'BinaryStatus']

import pandas as pd
import numpy as np

# corelation can be used to investigate which features are important
# get correlations of each features in dataset
corrmat = df_ohe.corr(method='pearson')
top_corr_features = corrmat.index
plt.figure(figsize=(35, 35))
plt.title('Correlation of the features in the dataset')
# plot heat map
g = sns.heatmap(df_ohe[top_corr_features].corr(), annot=True, cmap="RdYlGn")

# Feature extraction
num_features = 3
model = LogisticRegression(max_iter=300)
rfe = RFE(estimator=model, n_features_to_select=num_features)
fit = rfe.fit(df_ohe[X], Y)
print("Num Features:\n %s" % (fit.n_features_))
print("Selected Features:\n %s" % (fit.support_))
print("Feature Ranking:\n %s" % (fit.ranking_))

numpy_data = np.array([X, fit.support_, fit.ranking_])
feature_info = {'Feature': X, 'Support': fit.support_, 'Rank': fit.ranking_}
df_features = pd.DataFrame(data=feature_info, columns=["Feature", "Support", "Rank"])
print('All Features sorted in order of rank:\n', df_features.sort_values(by=['Rank']))
df_top_features = df_features[df_features["Rank"] == 1]
print(f'\n\nTop {num_features} features:\n ')
print(df_top_features)

'''
────────────────────────────── ❝ 2.3 ❞ ──────────────────────────────
   Train, test data splitting – use numpy, sci-kit learn.
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Check all correlations. Here it take longer time to execute 
graphDf = sns.pairplot(df_ohe)
graphDf.fig.suptitle("Corelation between all numeric columns before normalization", y=1.05)
'''
from sklearn import preprocessing

# Get column names first
# these are the features we will be focused on
feature_names = df_ohe.columns  # TO CONTINUE WITH SELECTEDD
'''feature_names = ['Occurrence_DayOfWeek', 'Occurrence_DayOfYear', 'Occurrence_Hour', 
            'Report_DayOfWeek', 'Report_DayOfYear', 'Report_Hour', 'Division',    
            'NeighbourhoodName','Hood_ID', 'Location_Type', 'Premises_Type' , 
            'Bike_Type', 'BinaryStatus','Primary_Offence' ]'''

# Standardization of a dataset is a common requirement for many machine learning estimators:
# they might behave badly if the individual features do not more or less look like
# standard normally distributed data
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_ohe[feature_names])
scaled_df = pd.DataFrame(scaled_df, columns=feature_names)
print(scaled_df.head())
print(scaled_df['BinaryStatus'].describe())
print(scaled_df.dtypes)

'''Normalize the data _____________________________________________________________________________________'''

# creating a new dataframe with the specified columns
df_new = scaled_df[feature_names].copy()

# Normalize the data in order to apply clustering
df_norm = (df_new - df_new.min()) / (df_new.max() - df_new.min())
df_norm.head()

# from sklearn import datasets
k = 3
model = KMeans(n_clusters=k)
model.fit(df_norm[feature_names])

model.labels_

# Append the clusters to each record on the dataframe, i.e. add a new column for clusters
md = pd.Series(model.labels_)
df_norm['Cluster'] = md
df_norm.head(10)

# get value counts for each cluster
df_norm['Cluster'].value_counts()

# find the final cluster's centroids for each cluster
model.cluster_centers_

# Calculate the J-scores The J-score can be thought of as the sum of the squared distance between points and cluster centroid for each point and cluster.
# For an efficient cluster, the J-score should be as low as possible.
model.inertia_

# export cluster data
df_norm.to_csv(path + 'Bicycle_Theft_Data_WithClusters.csv', index=False)

# Check all correlations. Here it take longer time to execute
graphDf = sns.pairplot(df_norm)
graphDf.fig.suptitle("Corelation between all numeric columns after normalization", y=1.05)

'''
#update as some columns were removed during feature selection

#plot clusters on chart
plt.title('Scatter of Clusters for Report_DayOfWeek vs BinaryStatus')
sns.scatterplot(x='Report_DayOfWeek', y='BinaryStatus', hue='Cluster', data=df_norm)

plt.title('Scatter of Clusters for Hours_Between vs BinaryStatus')
sns.scatterplot(x='Hours_Between', y='BinaryStatus', hue='Cluster', data=df_norm)

plt.title('Scatter of Clusters for Cost_ofBike vs BinaryStatus')
sns.scatterplot(x='Cost_of_Bike', y='BinaryStatus', hue='Cluster', data=df_norm)


#let us plot a histogram for the clusters
plt.hist(df_norm['Cluster'])
plt.title('Histogram of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')

# plot a scatter 
plt.title('Scatter plot of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.scatter(df_norm['Cluster'],df_norm['Hours_Between'])
plt.scatter(df_norm['Cluster'],df_norm['Report_DayOfWeek'])
plt.scatter(df_norm['Cluster'],df_norm['Occurrence_DayOfWeek'])
plt.scatter(df_norm['Cluster'],df_norm['Report_Month'])
plt.scatter(df_norm['Cluster'],df_norm['Occurrence_Month'])
plt.scatter(df_norm['Cluster'],df_norm['Cost_of_Bike'])
'''
# Remove clustering by dropping cluster column
df_norm = df_norm.drop(['Cluster'], axis=1)

'''Test & Train the data _____________________________________________________________________________________'''
from sklearn.linear_model import LogisticRegression

dependent_variable = 'BinaryStatus'
# Another way to split the features
x = scaled_df[scaled_df.columns.difference([dependent_variable])]
x.dtypes
y = scaled_df[dependent_variable]
# convert the class back into integer
y = y.astype(int)
# Split the data into train test
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2)
# build the model
lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)
# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score

score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print('==> ==> ==> The score of the 10 fold run is: ', score)

testY_predict = lr.predict(testX)
testY_predict.dtype
# print(testY_predict)
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

labels = y.unique()
print(labels)
print("==> ==> ==> Accuracy: ", metrics.accuracy_score(testY, testY_predict), '%')
# Print the confusion matrix
from sklearn.metrics import confusion_matrix

print("Confusion matrix Before normalization\n", confusion_matrix(testY, testY_predict, labels=labels))

# plot a heathmap for the confusion matrix
ax = plt.axes()
plt.figure(figsize=(10, 9))
sns.heatmap(data=confusion_matrix(testY, testY_predict, labels=labels), annot=True, cmap='Spectral', linewidth=0.5,
            ax=ax)
ax.set_title("Heatmap showing confusion matrix results", y=1.05)

# Dump pkl model and columns
import joblib

joblib.dump(lr, path + 'imbalanced-model_lr2.pkl')
print("imbalanced-Model dumped!")
model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, path + 'imbalanced-model_columns.pkl')
print("imbalanced-Models columns dumped!")

# Normalized column stats
df_norm.columns.values
pd.set_option('display.max_columns', 4)
pd.set_option('display.width', 200)
print(df_norm.groupby("BinaryStatus").mean())
print(df_norm.groupby("BinaryStatus").median())
print(df_norm.groupby("BinaryStatus").corr())

# Heatmap of corelation between all variables
ax = plt.axes()
ax.set_title("Heatmap of corelation between all variables", y=1.05)
sns.heatmap(df_norm.corr(), annot=True, cmap='coolwarm', linewidth=0.5, ax=ax)

# Heatmap of corelation between Predictors, weighted by BinaryStatus grouping
ax = plt.axes()
ax.set_title("Heatmap of corelation between Predictors, weighted by status grouping", y=1.05)
sns.heatmap(df_norm.groupby("BinaryStatus").corr(), annot=True, cmap='coolwarm', linewidth=0.5, ax=ax)

'''
#update as some columns were removed during feature selection

df_norm['Occurrence_Month'].value_counts()
df_norm['Occurrence_DayOfWeek'].value_counts()
df_norm['Report_Month'].value_counts()
df_norm['Report_DayOfWeek'].value_counts()
df_norm["Hours_Between"].value_counts()
df_norm['Cost_of_Bike'].value_counts()
'''

'''_______________________________________________________________________________________________________'''

'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++TESTING...imbalanced'''
print('Normalized but imbalanced:\n', df_norm['BinaryStatus'].value_counts())

'''
────────────────────────────── ❝ 2.4 ❞ ──────────────────────────────
   Managing imbalanced classes if needed. 
   Reference: https://elitedatascience.com/imbalanced-classes
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
'''

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# [STOLEN] = 1, [RECOVERED] = 0
# Separate majority and minority classes
df_majority = df_norm[df_norm.BinaryStatus == 1]
df_minority = df_norm[df_norm.BinaryStatus == 0]

'''
min_count = len(df_minority.index)
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=min_count,     # to match minority class
                                 random_state=123) # reproducible results
 
 

'''
max_count = len(df_majority.index)
# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=max_count,  # to match majority class
                                 random_state=123)  # reproducible results

'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++balanced and normalized dataset'''

# Combine majority class with upsampled minority class
df_norm_bal = pd.concat([df_majority, df_minority_upsampled])

# Combine minority class with downsampled majority class
# df_norm_bal = pd.concat([df_majority_downsampled, df_minority])


# Display new class counts
df_norm_bal.BinaryStatus.value_counts()

# Separate input features (X) and target variable (y)
y = df_norm_bal.BinaryStatus
X = df_norm_bal.drop('BinaryStatus', axis=1)

# Train model
clf_2 = LogisticRegression().fit(X, y)

# Predict on training set
pred_y_2 = clf_2.predict(X)

# Is our model still predicting just one class?
print(f'==> ==> ==> Prediction Options: {np.unique(pred_y_2)}')
# [0 1]

# How's our accuracy?
print(f'==> ==> ==> Accuracy: {accuracy_score(y, pred_y_2)}')

'''

╔═══════════════════════════════════════════════════════════════════════════════╗
✧･ﾟ: *✧･ﾟ:* 　⋆﹥━━━━━━━━━ 3. Predictive Model Building ━━━━━━━━━﹤⋆　 *:･ﾟ✧*:･ﾟ✧
╚═══════════════════════════════════════════════════════════════════════════════╝

'''

'''
─────────────────────────────── ❝ 3.1 ❞ ──────────────────────────────
  Use logistic regression and decision trees as a minimum– 
  use scikit learn.
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
'''

'''Test & Train the data _____________________________________________________________________________________'''

# Split the data into train test
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
# build the model
lr = LogisticRegression(solver='lbfgs')
lr.fit(X, y)

print('X columns: ', X.columns.values)
print(X.info())

print('y columns: ', y)

'''

╔═════════════════════════════════════════════════════════════════════════════════╗
✧･ﾟ: *✧･ﾟ:* 　⋆﹥━━━━━━━━━ 4. Model Scoring And Evaluation ━━━━━━━━━﹤⋆　 *:･ﾟ✧*:･ﾟ✧
╚═════════════════════════════════════════════════════════════════════════════════╝

'''

# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score

score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print('\n\n________________________________________________________________________________')

print("\n==> ==> ==> The score of the 10 fold run after handling imbalanced data is: ", score,
      " or %d%%" % (score * 100))

testY_predict = lr.predict(testX)
testY_predict.dtype
print(testY_predict)

'''
─────────────────────────────── ❝ 4.1 ❞ ──────────────────────────────
  Present results as scores, confusion matrices, and ROC - 
  use sci-kit learn.
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
'''
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

labels = y.unique()
print(labels)
accuracy = metrics.accuracy_score(testY, testY_predict)
print("\==> ==> ==> nAccuracy after handling imbalanced data is: ", accuracy, " or %d%%" % (accuracy * 100))

# Let us print the confusion matrix
from sklearn.metrics import confusion_matrix

print("Confusion matrix after handling imbalanced data\n", confusion_matrix(testY, testY_predict, labels=labels))
print('________________________________________________________________________________\n\n')
# plot a heathmap for the confusion matrix
ax = plt.axes()
plt.figure(figsize=(4, 4))
sns.heatmap(data=confusion_matrix(testY, testY_predict, labels=labels), annot=True, cmap='Spectral', linewidth=0.5,
            ax=ax)
ax.set_title("Heatmap showing confusion matrix after handling imbalanced data", y=1.05)

'''
─────────────────────────────── ❝ 4.2 ❞ ──────────────────────────────
  Select and recommend the best performing model.
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
'''
import joblib

joblib.dump(lr, path + 'model_lr2.pkl')
print("balanced-Model dumped!")

model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, path + 'model_columns.pkl')
print("balanced-Models columns dumped!")

'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++TESTING...BALANCED'''

'''
 python "./Grp_Assign2.py"
'''

print('Balance and normalized:\n', df_norm_bal['BinaryStatus'].value_counts())
# Cannot take a larger sample than population when 'replace = False'
test_data = df_norm_bal.sample(n=10, replace=True)
print(test_data)
test_predictions = test_data['BinaryStatus']
test_data = test_data.drop(['BinaryStatus'], axis=1)

print(test_data)
print(test_predictions)

# print(test_data.to_json(orient='records', lines=True).splitlines())
# json_testData = test_data.to_json(orient='records', lines=True)
json_testData = test_data.to_json(orient='records')
# print(json_testData)

# json_testPredictions = test_predictions.to_json(orient='records', lines=True).splitlines()
json_testPredictions = test_predictions.to_json(orient='records')
# print(json_testPredictions)


json_testObject = {"test_data": json_testData, "expected_results": json_testPredictions}

print(json_testObject)

'''

╔═════════════════════════════════════════════════════════════════════════════════╗
✧･ﾟ: *✧･ﾟ:* 　⋆﹥━━━━━━━━━ 5. Model Scoring And Evaluation ━━━━━━━━━﹤⋆　 *:･ﾟ✧*:･ﾟ✧
╚═════════════════════════════════════════════════════════════════════════════════╝

'''

'''
─────────────────────────────── ❝ 5.1 ❞ ──────────────────────────────
  Using a flask framework arrange to turn your selected 
  machine-learning model into an API.
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
'''

'''
─────────────────────────────── ❝ 5.2 ❞ ──────────────────────────────
  Using the pickle module, arrange for Serialization & Deserialization 
  of your model.
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
'''

'''
─────────────────────────────── ❝ 5.3 ❞ ──────────────────────────────
  Build a client to test your model API service. Use the test data, 
  which was not previously used to train the module. You can use simple 
  Jinja HTML templates with or without Java script, REACT or any other 
  technology but at minimum use POSTMAN Client API.
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
'''

"""
"""
from flask import Flask, request, jsonify
import traceback, joblib

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            # print(json_['expected_results'])

            expected_results = request.json['expected_results']
            count_recovered = expected_results.count(0)
            count_stolen = expected_results.count(1)

            query = pd.get_dummies(pd.DataFrame(json_['test_data']))
            query = query.reindex(columns=model_columns, fill_value=0)

            # print(query)
            prediction = lr.predict(query).tolist()
            count_0 = prediction.count(0)
            count_1 = prediction.count(1)

            # dataset = pd.DataFrame({'test data': request.json['test_data'], 'expected values': expected_results, 'predicted values': prediction})
            dataset = pd.DataFrame({'expected values': expected_results, 'predicted values': prediction})
            json_dataset = dataset.to_json(orient='records', lines=True).splitlines()

            matrix = confusion_matrix(expected_results, prediction, labels=[0, 1])
            json_truePositive = int(matrix[0][0])
            json_falsePositive = int(matrix[0][1])
            json_falseNegative = int(matrix[1][0])
            json_trueNegative = int(matrix[1][1])

            accuracy = metrics.accuracy_score(expected_results, prediction)
            print("\nAccuracy of the prediction is: ", accuracy, " or %d%%" % (accuracy * 100))

            json_matrix = json.dumps(matrix.tolist())

            # [STOLEN] = 1, [RECOVERED] = 0
            out_json = {
                'Values predicted': prediction,
                'Values expected': expected_results,
                'Bicycles Predicted Recovered': count_0,
                'Bicycles Predicted Stolen': count_1,
                'Bicycles Actually Recovered': count_recovered,
                'Bicycles Actually Stolen': count_stolen,
                'Comparison': json_dataset,
                'Accuracy of the prediction': accuracy,
                "The Confusion matrix": json_matrix,
                "Total True Positive": json_truePositive,
                "Total False Positive": json_falsePositive,
                "Total True Negative": json_trueNegative,
                "Total False Negative": json_falseNegative
            }
            print(out_json)
            return jsonify(out_json)

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Model Required')
        return ('Model Not Found')


if __name__ == '__main__':
    print("Loading Model...")
    lr = joblib.load(path + 'model_lr2.pkl')
    print('...Model loaded!')

    print("Loading Model Columns...")
    model_columns = joblib.load(path + 'model_columns.pkl')
    print('....Model columns loaded!')

    app.run(port=12345, debug=True)

'''

╔═══════════════════════════════════════════════════════════════════════╗
✧･ﾟ: *✧･ﾟ:* 　⋆﹥━━━━━━━━━ 6. Prepare A Report ━━━━━━━━━﹤⋆　 *:･ﾟ✧*:･ﾟ✧
╚═══════════════════════════════════════════════════════════════════════╝

'''

'''
──────────────────────────────────────────────────────────────── ❝ 6.1 ❞ ───────────────────────────────────────────────────────────────
  Explain your project and detail all the assumptions and constraints 
  you applied in the following sections:

    1. Executive summary (to be written once nearing the end of the project work, should describe the problem/solution and key findings);
    2. Overview of your solution (to be written once nearing the end of the project work);
    3. Data exploration and findings (dataset field descriptions, graphs, visualizations, tools, and libraries used, etc.);
    4. Feature selection (tools and techniques used, results of different combinations, etc.);
    5. Data modeling (data cleaning strategy, results of data cleaning, data wrangling techniques, assumptions and constraint); and
    6. Model building (train/ test data, sampling, algorithms tested, results: confusion matrixes, etc.).
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
'''
# -->just give a simple summary of the project
