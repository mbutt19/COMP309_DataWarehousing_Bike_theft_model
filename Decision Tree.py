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

path = "./"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
data_mustafa_i = pd.read_csv(fullpath,sep=',',low_memory=False)
pd.options.mode.chained_assignment = None

# Data Assesment
# print(data_mustafa_i)
print(data_mustafa_i.head(5))
print(data_mustafa_i.columns.values)
print(data_mustafa_i.shape)
print(data_mustafa_i.describe())
print(data_mustafa_i.info()) 
print(len(data_mustafa_i)-data_mustafa_i.count())
#print(data_mustafa_i['Status'].value_counts())
print(data_mustafa_i['Status'].value_counts())

# Plotting variables
#data_mustafa_i["Status"].value_counts().plot.pie(autopct='%.2f')
#sns.pairplot(data_mustafa_i, hue="Bike_Type")
#sns.pairplot(data_mustafa_i, hue="Premise_Type")
#sns.pairplot(data_mustafa_i, hue="Status")

# Calculate time between reporting and occurrence (in Days & Hours)
data_mustafa_i['Days_Between'] = [(data_mustafa_i["Report_DayOfYear"][i] - data_mustafa_i["Occurrence_DayOfYear"][i]) 
                                  if (data_mustafa_i["Report_DayOfYear"][i]>=data_mustafa_i["Occurrence_DayOfYear"][i]) 
                                  else (data_mustafa_i["Report_DayOfYear"][i]-data_mustafa_i["Occurrence_DayOfYear"][i])+365 
                                         for i in range(len(data_mustafa_i["Occurrence_DayOfYear"]))]
data_mustafa_i["Total_Hours_Between"] = data_mustafa_i["Report_Hour"]

for i in range(len(data_mustafa_i["Report_Date"])):
    data_mustafa_i["Total_Hours_Between"][i] = (pd.to_datetime(data_mustafa_i["Report_Date"][i]) - pd.to_datetime(data_mustafa_i["Occurrence_Date"][i])).total_seconds()/3600 
#data_mustafa_i["Report_Date"] = time.mktime(datetime.datetime.strptime(data_mustafa_i["Report_Date"], "%Y/%m/%d %H").timetuple())

# Splitting the predictor and target variables and create dataframe with only selected features
target=["Status"]
predictors=["Premises_Type", "Days_Between", 'Bike_Type', 'Cost_of_Bike',"Total_Hours_Between"]
include = predictors
selected_data_raw = data_mustafa_i[include]
print("Selected Data:\n",selected_data_raw.info())

# Categorical handling
selected_data = pd.get_dummies(selected_data_raw, columns=["Premises_Type", 'Bike_Type'], dummy_na=True)

# Standardize
predictors = selected_data.columns.values
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(selected_data)
scaled_df = pd.DataFrame(scaled_df, columns=predictors)
scaled_df["Status"] = data_mustafa_i["Status"]
scaled_df.fillna(0,inplace=True)
print(scaled_df.info())
print(scaled_df.describe())

# Correct imbalance in status column
scaled_df = scaled_df[scaled_df.Status != "UNKNOWN"]
scaled_df['Status'] = [1 if b=='STOLEN' else 0 for b in scaled_df.Status]
# Separate majority and minority classes
df_majority = scaled_df[scaled_df.Status==1]
df_minority = scaled_df[scaled_df.Status==0]
# Upsample minority class
plt.pie(data_mustafa_i["Status"].value_counts(), labels=data_mustafa_i["Status"].unique(), autopct='%.2f', pctdistance=.88, radius=1.5)
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123) # reproducible results
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
scaled_df = resample(df_upsampled, 
                                 replace=False,     # sample with replacement
                                 n_samples=len(scaled_df["Status"]),    # to match majority class
                                 random_state=123) # reproducible results

print(scaled_df['Status'].value_counts())
"""
# Correct imbalance in status column
#plt.pie(data_mustafa_i["Status"].value_counts(), labels=data_mustafa_i["Status"].unique(), autopct='%.2f', pctdistance=.88, radius=1.5)
#data_mustafa_i['Status'] = [1 if b=='STOLEN' else if b=='UNKNOWN' 2 else 0 for b in data_mustafa"_i.Status]
data_mustafa_i = data_mustafa_i[data_mustafa_i.Status != "UNKNOWN"]
data_mustafa_i['Status'] = [1 if b=='STOLEN' else 0 for b in data_mustafa_i.Status]
print(data_mustafa_i['Status'].value_counts())
sample = data_mustafa_i["Status"]

# Separate majority and minority classes
df_majority = data_mustafa_i[data_mustafa_i.Status==0]
df_minority = data_mustafa_i[data_mustafa_i.Status==1]
"""

# Splitting the dataset into train and test variables 
scaled_df['is_train'] = (np.random.uniform(0, 1, len(scaled_df)) <= .75)
# print(selected_data.head(5))

# Create two new dataframes, one with the training rows, one with the test rows
train, test = scaled_df[scaled_df['is_train']==True], scaled_df[scaled_df['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# Build decision tree with training data
dt_mustafa = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_mustafa.fit(train[predictors], train[target])

# Test the model and create confusion matrix
preds=dt_mustafa.predict(test[predictors])
print(pd.crosstab(test['Status'],preds,rownames=['Actual'],colnames=['Predictions']))

# Create dotfile to view on graphviz online and save as a PNG
from sklearn.tree import export_graphviz
with open('./DecisionTree.dot', 'w') as dotfile:
   export_graphviz(dt_mustafa, out_file = dotfile, feature_names = predictors,filled = True,
            rounded = True, proportion=True)
dotfile.close()

# Split the data using sklearn's train_test_split module instead
X=scaled_df[predictors]
Y=scaled_df[target]
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
print("TRAIN/TEST SET:\n", trainX.head(5),"===\n", trainY.head(5),"===\n", testX.head(5),"===\n", testY.head(5))
print("TrainX=",len(trainX),"TrainY=",len(trainY),"TestX=",len(testX),"TestY=",len(testY))

# Rebuild decision tree
dt1_mustafa = DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_split=20, random_state=99)
dt1_mustafa.fit(trainX,trainY)

# 10 fold cross validation using sklearn and all the data i.e validate the data 
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
score = np.mean(cross_val_score(dt1_mustafa, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print(score)
 
# Test the model using the testing data
testY_predict = dt1_mustafa.predict(testX)
testY_predict.dtype

# Import scikit-learn metrics module for accuracy calculation
labels = Y["Status"].unique()
print("TEST DATA:\nLabels: ",labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))

# Print the confusion matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=labels))

# Use Seaborn heatmaps to print confusion matrix 
cm = confusion_matrix(testY, testY_predict, labels=labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# Labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); 
ax.yaxis.set_ticklabels(labels);
plt.show()

# Decision Tree model will be built and run 15 times for more rigorous score
score_card = list()
accuracy_card = list()
cm_card = list()
for i in range(1, 16):
    # Rebuild decision tree
    dt1_mustafa = DecisionTreeClassifier(criterion='entropy',max_depth=i, min_samples_split=20, random_state=99)
    dt1_mustafa.fit(trainX,trainY)
    # 10 fold cross validation using sklearn and all the data i.e validate the data 
    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
    score = np.mean(cross_val_score(dt1_mustafa, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
    score_card.append(score)
    # Test the model using the testing data
    testY_predict = dt1_mustafa.predict(testX)
    testY_predict.dtype
    # Import scikit-learn metrics module for accuracy calculation
    labels = Y["Status"].unique()
    accuracy_card.append(metrics.accuracy_score(testY, testY_predict))   
    # Print the confusion matrix
    print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=labels))  
    # Use Seaborn heatmaps to print confusion matrix 
    cm = confusion_matrix(testY, testY_predict, labels=labels)
    cm_card.append(cm)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells 
    # Labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(labels); 
    ax.yaxis.set_ticklabels(labels);
    plt.show()
print(score_card)
print(accuracy_card)
#print(cm_card)

# Feature importance testing
fi = dt1_mustafa.feature_importances_
print("- - - - - FEATURE IMPORTANCE - - - - -")
for i in range(len(fi)):
    print(list(scaled_df.columns.values)[i], ": ", fi[i])
