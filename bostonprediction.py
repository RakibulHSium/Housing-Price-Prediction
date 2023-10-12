# -*- coding: utf-8 -*-
"""bostonprediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NVbId0Vz3OcLl6zJhUDqcr8-yqKfXKsV

<center>    
<h2>Algorithm Design Final Project</h2>
<h3>
Housing market prediction by machine learning algorithms</h3>
<h3>SIUM RAKIBUL HASAN</h3>

<br><h4> Prof. Zheng Yu </h4>
</center>
"""

## Importing required packages.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import time
import math
from random import uniform
from scipy.stats import  randint as sp_randint

from tqdm import tqdm


from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,ConfusionMatrixDisplay,precision_score,recall_score,f1_score,classification_report,roc_curve,plot_roc_curve,auc,precision_recall_curve,plot_precision_recall_curve,average_precision_score,mean_squared_error, r2_score, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression, ridge_regression, Ridge, Lasso
from sklearn.feature_selection import SequentialFeatureSelector

from lightgbm import LGBMClassifier, LGBMRegressor

import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # Import Random Forest Classifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:.2f}'.format

## Reading the data from the File.

dataset = pd.read_csv("/content/data2021-full.csv")
dataset.head(5)

dataset.shape

dataset.info()

"""## Data Preprocessing and EDA"""

duplicates = dataset[dataset.duplicated(['PID'])]
duplicates
print("The number of duplicates found in PID column:", duplicates.count())  # PID = 176

dataset.drop_duplicates(['PID'],keep= 'last', inplace=True)

duplicates = dataset[dataset.duplicated(['PID'])]

print("The number of duplicates found in PID column:", duplicates.count())

# Check the percentage and count of the missing data 

for col in dataset.columns:
    pct_missing = np.mean(dataset[col].isnull())
    print('{} \t: {}% \t: {}'.format(sum(dataset[col].isnull()), round(pct_missing*100),col))

index_names = ['PID','CM_ID','GIS_ID','ST_NUM','UNIT_NUM','MAIL_ADDRESSEE','LUC','LU', \
                       'CD_FLOOR','RES_FLOOR','RES_UNITS','COM_UNITS','LAND_SF', 'LU_DESC',\
                       'LAND_VALUE','YR_REMODEL','STRUCTURE_CLASS','ROOF_STRUCTURE', \
                       'ROOF_COVER','INT_WALL','INT_COND','BED_RMS','HLF_BTH','FULL_BTH',
                       'BDRM_COND', 'BTHRM_STYLE2','BTHRM_STYLE3','KITCHEN_STYLE2', \
                       'KITCHEN_STYLE3', 'HEAT_FUEL', 'FIRE_PLACE', 'PlUMBING', \
                       'NUM_PARKING', 'PROP_VIEW', 'CORNER_UNIT', \
                       'OWNER', 'OWN_OCC', 'ST_NAME','MAIL_ADDRESS',
                        'MAIL_CITY', 'MAIL_ZIPCODE','OVERALL_COND','KITCHEN_STYLE1']

    
    
# drop these row 
# from dataFrame
BosProp = dataset.drop(index_names, inplace = False, axis='columns')

BosProp.head(5)

# Check the percentage and count of the missing data 

for col in BosProp.columns:
    pct_missing = np.mean(BosProp[col].isnull())
    print('{} \t: {}% \t: {}'.format(sum(BosProp[col].isnull()), round(pct_missing*100),col))

BosProp.dropna(subset=['AC_TYPE'],inplace=True)

# Check the percentage and count of the missing data 

for col in BosProp.columns:
    pct_missing = np.mean(BosProp[col].isnull())

    print('{} \t: {}% \t: {}'.format(sum(BosProp[col].isnull()), round(pct_missing*1000),col))

# Droping the rows with missing values.
BosProp.dropna(inplace=True)

# Check the percentage and count of the missing data 

for col in BosProp.columns:
    pct_missing = np.mean(BosProp[col].isnull())
    print('{} \t: {}% \t: {}'.format(sum(BosProp[col].isnull()), round(pct_missing*100),col))

BosProp.shape #(122570, 32) to (128825, 22)

## Counting all the states values as we focus on MA (Boston)  only.

BosProp["MAIL_STATE"].value_counts()

## Keeping only MA values as we focus on Boston only.


index_names = BosProp[ BosProp['MAIL_STATE'] != 'MA' ].index
  
# drop these row indexes
# from dataFrame
BosProp.drop(index_names, inplace = True) ## (124101, 22)

## As all the values are MA, it woudl not make sense to perform predective anlytics on it.     
    
# drop these row 
# from dataFrame
BosProp.drop('MAIL_STATE', inplace = True, axis='columns')

### Converting currency to float.
### The currency was in $719,400.00 format. we need to conveert it to 719400.00 to perform perform predective anlytics on it.

for col in ['BLDG_VALUE','TOTAL_VALUE','GROSS_TAX']:
    BosProp[col]=BosProp[col].replace({'\$': '', ',': ''}, regex=True,).astype(float)

BosProp.head(3)

## To detect counts and destribution among categorical values.


for col in ['CITY','NUM_BLDGS','BLDG_TYPE','YR_BUILT','EXT_FINISHED', 'EXT_COND','KITCHEN','TT_RMS','BTHRM_STYLE1','KITCHEN_TYPE','HEAT_TYPE','AC_TYPE']:
    print("# # # # #  ",col,"  # # # # #")
    display(BosProp[[col]].value_counts())

## Now that we have all data cleaned, We performed describe function to know the count, unique, max, min and other paterns.
BosProp.describe(include="all",datetime_is_numeric=True)



#BosProp['TOTAL_VALUE'].sum()/123921

## A Bar plot to see, YR_BUILT Desctribution

sns.displot(data=BosProp['YR_BUILT'])

## A Box plot to see, YR_BUILT outliars and quartraints 

sns.boxplot(data=BosProp['YR_BUILT'])

## A Box plot to see, TOTAL_VALUE outliars and quartraints 

sns.boxplot(data=BosProp['TOTAL_VALUE'])

## Displaying outliars in TOTAL_VALUE that is values above 2000000


BosProp['TOTAL_VALUE'][BosProp.TOTAL_VALUE>2100000]

## Removing outliars in TOTAL_VALUE that is values above 2000000

index_names = BosProp[ BosProp.TOTAL_VALUE>2000000 ].index
  
# drop these row indexes
# from dataFrame
BosProp.drop(index_names, inplace = True)
BosProp.TOTAL_VALUE.value_counts()

## A Box plot to see, TOTAL_VALUE outliars and quartraints 


sns.displot(data=BosProp['TOTAL_VALUE'])

## A Box plot to see, GROSS_AREA outliars and quartraints 


sns.boxplot(data=BosProp['GROSS_AREA'])

#BosProp[BosProp['GROSS_AREA']<25000]


## Removing outliars in GROSS_AREA that is values above 25000


sns.boxplot(data=BosProp[BosProp['GROSS_AREA']<25000]['GROSS_AREA'])

## A Bar plot to see, GROSS_AREA Desctribution


sns.displot(data=BosProp['GROSS_AREA'])

## A Box plot to see, GROSS_TAX outliars and quartraints 



sns.boxplot(data=BosProp['GROSS_TAX'])

## A Bar plot to see, GROSS_TAX Desctribution


sns.displot(BosProp['GROSS_TAX'])

## A Box plot to see, LIVING_AREA outliars and quartraints 


sns.boxplot(data=BosProp['LIVING_AREA'])

## A Bar plot to see, LIVING_AREA Desctribution


sns.displot(BosProp['LIVING_AREA'])

## A Box plot to see, BLDG_VALUE outliars and quartraints 


sns.boxplot(data=BosProp['BLDG_VALUE'])

## A Bar plot to see, TOTAL_VALUE Desctribution


sns.displot(BosProp['TOTAL_VALUE'])



## A Bar plot to see, Desctribution, count of unique values in each feature. 


for i in ['CITY','BLDG_TYPE','EXT_FINISHED','EXT_COND','BTHRM_STYLE1','KITCHEN_TYPE','HEAT_TYPE','AC_TYPE']:
    fig, ax1 = plt.subplots(figsize=(6,4))
    graph = sns.countplot(ax=ax1,x=i, data=BosProp)
    graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x()+p.get_width()/2., height + 0.5,height ,ha="center")

plt.show()

77920/124101*100

## Checking if we have all the data in required format, not-null count, total records and features

BosProp.info()

# Checking the mean property value with respect to the city.

(BosProp[['TOTAL_VALUE','CITY']].groupby(['CITY']).mean()*0.1).nlargest(5, 'TOTAL_VALUE')

# Checking the min property value with respect to the city.


BosProp[['TOTAL_VALUE','CITY']].groupby(['CITY']).min().nlargest(5, 'TOTAL_VALUE')#.groupby(['TOTAL_VALUE']).mean()

# Checking the max property value with respect to the city.


BosProp[['TOTAL_VALUE','CITY']].groupby(['CITY']).max().nsmallest(5, 'TOTAL_VALUE')#.groupby(['TOTAL_VALUE']).mean()

nullseries = BosProp.isnull().sum()
display(nullseries)
print(nullseries[nullseries > 20])  ## Double checking if any missing values are leftout.



BosProp.keys()

for col in ['CITY', 'ZIPCODE', 'BLDG_TYPE', 'EXT_FINISHED', 'EXT_COND', 'KITCHEN', 'BTHRM_STYLE1', 'KITCHEN_TYPE', 'HEAT_TYPE', 'AC_TYPE'] :
    print("# # # # #  ",col,"  # # # # #")
    display(BosProp[[col]].value_counts())

sns.countplot(data=BosProp, x='CITY')
plt.xticks(rotation=90)
plt.show()

BosProp.select_dtypes(include='object').keys()

plt.figure(figsize = (20,20))

sns.jointplot(x='GROSS_AREA', y='TOTAL_VALUE', data=BosProp)

plt.show()
#ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)

fltBos = BosProp[BosProp['GROSS_AREA']>20000]

# fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(15, 8))
# index = 0
# axs = axs.flatten()
# for k,v in fltBos.select_dtypes(include='number').items():
#     sns.boxplot(y=k, data=fltBos, ax=axs[index])
#     index += 1
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

#fltBos.hist(figsize=(8,7))
plt.show()

# import seaborn as sns
# from matplotlib import pyplot as plt
# sns.set_style("ticks")
# sns.pairplot(BosProp,hue = 'ZIPCODE',diag_kind = "kde",kind = "scatter",palette = "husl")
# plt.show()

""" ## Feature Engineering"""

BosProp

## Converting Int, float values into string. As they are categorical values. 

BosProp['ZIPCODE'] = BosProp['ZIPCODE'].apply(str)
BosProp['YR_BUILT'] = BosProp['YR_BUILT'].apply(str)

# limit to categorical data using df.select_dtypes() Creating a sub dataset to work with.
Cat = BosProp.select_dtypes(include=[object])
Cat.head(3)

# check original shape
Cat.shape

# Categorical Columns. Rechecking if correct features are selected.
Cat.columns



# created a LabelEncoder object and fit it to each feature in X

# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()


# used apply() to apply le.fit_transform to all columns
CatToInt = Cat.apply(le.fit_transform)
CatToInt.head()

## Droping the acutal columns as we have encoded them as per out needs.

EnBosProp = BosProp.drop(Cat.columns, inplace = False, axis='columns')
EnBosProp.head(2)

## Combining the two dataframes, encoded one and acual numeric one into one dataset to work with. 

EnBosProp = pd.concat([EnBosProp, CatToInt], axis=1)
EnBosProp.head(2)

## Creating bins for BLDG_VALUE as the range is too large. 

print('The Min value is, ', EnBosProp.BLDG_VALUE.min(), ' and the max value is, ', EnBosProp.BLDG_VALUE.max())

EnBosProp['BLDG_VALUE_bins'] = pd.cut(EnBosProp.BLDG_VALUE, bins=5, labels=[1,2,3,4,5], include_lowest=True)

EnBosProp['BLDG_VALUE_bins'] = [int(x) for x in EnBosProp['BLDG_VALUE_bins']]

## Number of bins and count of values in that bin.
EnBosProp.BLDG_VALUE_bins.value_counts()

# Creating bins for TOTAL_VALUE as the range is too large. 

print('The Min value is, ', EnBosProp.TOTAL_VALUE.min(), ' and the max value is, ', EnBosProp.TOTAL_VALUE.max())


EnBosProp['TOTAL_VALUE_bins'] = pd.cut(EnBosProp.TOTAL_VALUE, bins=5, labels=[1,2,3,4,5], include_lowest=True)
EnBosProp['TOTAL_VALUE_bins'] = [int(x) for x in EnBosProp['TOTAL_VALUE_bins']]

## Number of bins and count of values in that bin.

EnBosProp.TOTAL_VALUE_bins.value_counts()

## Droping the acual values as we have created the bins

index_names = ['TOTAL_VALUE','BLDG_VALUE'] 
EnBosProp = EnBosProp.drop(index_names, inplace = False, axis='columns')

## EnBosProp

EnBosProp.columns ## Double checking if all the columns are the same that we intend to use.

from statsmodels.stats.outliers_influence import variance_inflation_factor

## Performing VIF (variance inflation factor) to track multicollinearity 

# the independent variables set
X = EnBosProp.drop(['TOTAL_VALUE_bins'], axis=1)

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)

## Removing the BLDG_SEQ feature as it has high VIF values

index_names = ['BLDG_SEQ'] 
EnBosProp = EnBosProp.drop(index_names, inplace = False, axis='columns')

from statsmodels.stats.outliers_influence import variance_inflation_factor

## Performing VIF (variance inflation factor) again to track multicollinearity in remaining features.


# the independent variables set
X = EnBosProp.drop(['TOTAL_VALUE_bins'], axis=1)

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)

## Removing the GROSS_AREA','GROSS_TAX','TT_RMS','LIVING_AREA', and 'ZIPCODE' feature as it has high VIF values


index_names = ['GROSS_AREA','GROSS_TAX','TT_RMS','LIVING_AREA','ZIPCODE'] 
EnBosProp = EnBosProp.drop(index_names, inplace = False, axis='columns')

from statsmodels.stats.outliers_influence import variance_inflation_factor


## Performing VIF (variance inflation factor) again to track multicollinearity in remaining features.

# the independent variables set
X = EnBosProp.drop(['TOTAL_VALUE_bins'], axis=1)

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)

## Removing the NUM_BLDGS, YR_BUILT feature as it has high VIF values


index_names = ['NUM_BLDGS','YR_BUILT'] 
EnBosProp = EnBosProp.drop(index_names, inplace = False, axis='columns')

from statsmodels.stats.outliers_influence import variance_inflation_factor

## Performing VIF (variance inflation factor) again to track multicollinearity in remaining features.


# the independent variables set
X = EnBosProp.drop(['TOTAL_VALUE_bins'], axis=1)

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)  ## We found all the vlaues are below the decided threshold 10.

EnBosProp

## Checking if we have all the data in required format, not-null count, total records and features


EnBosProp.info()

"""## Predictive Data Modeling"""

## Creating the independent features.

X= EnBosProp.drop(['TOTAL_VALUE_bins'], axis=1)

## Creating the dependent features.

y=EnBosProp['TOTAL_VALUE_bins']

## Spliting the data into test and train, with 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6040)

print('X_train: ',X_train.shape)
print('X_test: ',X_test.shape)
print('y_train: ',y_train.shape)
print('y_test: ',y_test.shape)

# Creating a dataframe to store model accuracy.

Models = []

"""#### Linear regression. """

model = LinearRegression()
#model.fit(X_train, y_train)

start_time = time.time()


model = LinearRegression().fit(X_train, y_train)
print("---Time taken to fit Model:  %s seconds ---" % (time.time() - start_time))
r_sq = model.score(X_test, y_test)


print('coefficient of determination:', r_sq)

mod = sm.OLS(y_train, X_train)

res = mod.fit()

print(res.summary())

pval = pd.DataFrame(res.pvalues, columns=['pval'])
coef = pd.DataFrame(res.params, columns= ['coef'])
combine = pd.concat([pval, coef], axis = 1)
combine
combine[(combine['pval'] < 0.05)==True]

# performing predictions on the test datdaset
pred = model.predict(X_test)
prediction = list(map(round, pred))

# confusion matrix
cm = confusion_matrix(y_test, prediction)
print("== Linear Regression ==")
print ("Confusion Matrix : \n", cm)
accuracy = accuracy_score(y_test, prediction)

Models.append(['Linear Regression', accuracy])

# accuracy score of the model
print('Test accuracy = ', accuracy)
print('Mean Squared Error = ', mean_squared_error(y_test, prediction))

print('Linear Regression Confusion Matrix with labels\n\n');

dictsplay_labels=y.unique()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dictsplay_labels)

disp = disp.plot()

plt.show()



"""### Random Forest"""

regr = RandomForestRegressor(max_depth=2, random_state=0)

start_time = time.time()
regr.fit(X_train, y_train)
print("---Time taken to fit Model:  %s seconds ---" % (time.time() - start_time))

y_pred=regr.predict(X_test)

regr.score(X_test, y_test)

accuracy = regr.score(X_test, y_test)
Models.append(['Random Forest',accuracy])
accuracy

# performing predictions on the test datdaset
pred = regr.predict(X_test)
prediction = list(map(round, pred))
cm_regr=confusion_matrix(y_test, prediction)
cm_regr

print('Random Forest Confusion Matrix with labels\n\n');

dictsplay_labels=y.unique()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_regr, display_labels=dictsplay_labels)

disp = disp.plot()

plt.show()

importances = list(regr.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train.columns, importances)]
feature_importances = sorted(feature_importances, key = lambda X_train: X_train[1], reverse = True) 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

## Extracting the best features from the modle using Stepwise Selection

sfs = SequentialFeatureSelector(regr, n_features_to_select=3)
sfs.fit(X_train, y_train)
sfs.get_feature_names_out()

"""#### LightGBM Regression

"""



start_time = time.time()
lgb_model = LGBMRegressor(subsample=0.9).fit(X_train, y_train)
print("---Time taken to fit Model:  %s seconds ---" % (time.time() - start_time))

#best_params = lgb_random.best_params_
#print(best_params)

accuracy = lgb_model.score(X_train, y_train)
Models.append(['LightGBM',accuracy]) # LightGBM Regression accuracy
accuracy



# performing predictions on the test datdaset
pred = lgb_model.predict(X_test)
prediction = list(map(round, pred))

# confusion matrix
cm = confusion_matrix(y_test, prediction)
print("== LightGBM Regression ==")
print ("Confusion Matrix : \n", cm)

# accuracy score of the model
print('Mean Squared Error = ', mean_squared_error(y_test, prediction))

print('LightGBM Regression Confusion Matrix with labels\n\n');

dictsplay_labels=y.unique()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dictsplay_labels)

disp = disp.plot()

plt.show()

lgb_model.feature_importances_

importances = list(lgb_model.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train.columns, importances)]
feature_importances = sorted(feature_importances, key = lambda X_train: X_train[1], reverse = True) 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

## Extracting the best features from the modle using Stepwise Selection

sfs = SequentialFeatureSelector(lgb_model, n_features_to_select=3)
sfs.fit(X_train, y_train)
sfs.get_feature_names_out()

"""### Model Regularization for Stable Sample Rollouts

#### Ridge regression
"""

start_time = time.time()
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
print("---Time taken to fit Model:  %s seconds ---" % (time.time() - start_time))

accuracy = ridge.score(X_train, y_train)
Models.append(['Ridge Regression',accuracy])# Ridge Regression accuracy
accuracy

## Extracting the best features from the modle using Stepwise Selection
sfs = SequentialFeatureSelector(ridge, n_features_to_select=2)
sfs.fit(X_train, y_train)
sfs.get_feature_names_out()

"""#### Lasso regression

"""

start_time = time.time()
lasso = Lasso(alpha=0.01).fit(X_train, y_train) 
print("---Time taken to fit Model:  %s seconds ---" % (time.time() - start_time))
pred_train_lasso = lasso.predict(X_train)

accuracy = lasso.score(X_train, y_train)
Models.append(['Lasso Regression',accuracy]) # Lasso Regression
accuracy

## Extracting the best features from the modle using Stepwise Selection

sfs = SequentialFeatureSelector(lasso, n_features_to_select=2)
sfs.fit(X_train, y_train)
sfs.get_feature_names_out()

"""### Model Comparison"""

## A Table to see, AllModels Accuracy 


AllModels = pd.DataFrame(Models,columns=['Model','Accuracy'])

AllModels.sort_values(by=['Accuracy'],inplace=True,ascending=False)
AllModels

## A Bar plot to see, AllModels Accuracy 

sns.barplot(data=AllModels, x='Accuracy', y='Model')
