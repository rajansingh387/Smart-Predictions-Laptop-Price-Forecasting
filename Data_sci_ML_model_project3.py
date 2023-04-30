# Import the libraries
 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

# Read the dataset

df = pd.read_csv("laptop_price_data.csv")
print(df.head())

# Check the shape of the dataset

print(df.shape) #->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> [(1302, 13)]

# Data Pre-processing
# 1) Handle the Null values

print(df.isnull().sum()) 
 
 #[5 rows x 13 columns]
 # Unnamed: 0     0
 # Company        0
 # TypeName       0
 # Ram            0
 # Weight         0
 # Price          0
 # Touchscreen    0
 # Ips            0
 # Cpu brand      0
 # HDD            0
 # SSD            0
 # Gpu brand      0
 # os             0
 # dtype: int64
 # NO null values 


# 2) Handle the Duplicates

print(df.duplicated().sum()) #->>>>>>>>>>>>>>>>>>>>>>>>>>> No duplicates are thier

# 3) Drop redundant Columns
# First let's check the columns names of the data set

print(df.columns)

#Index(['Unnamed: 0', 'Company', 'TypeName', 'Ram', 'Weight', 'Price',
#       'Touchscreen', 'Ips', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os']

print(df['Unnamed: 0'].nunique()) #->>>>>>>>>>>>>>>>>>>>>> 1302

# Now we drop this column

(df.drop('Unnamed: 0',axis=1,inplace=True))
print(df.columns)

# Here clearly we can see that the above mentioned column is dropped

# Index(['Company', 'TypeName', 'Ram', 'Weight', 'Price', 'Touchscreen', 'Ips',
#       'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os']

# 4) Check the data types
 
print(df.dtypes)

#Company         object
#TypeName        object
#Ram              int64
#Weight         float64
#Price          float64
#Touchscreen      int64
#Ips              int64
#Cpu brand       object
#HDD              int64
#SSD              int64
#Gpu brand       object
#os              object

# see columns Company,TypeName,Cpu brand,Gpu brand and os are of object data type as ML can't
# accept object datatype so we need to perform encoding -> so we encode these columns using
# pipeline technique before this we do 
# 1) EDA (Exploratory Data Analysis)
# 2) Correlation
# 3) handling outliers  


# EDA (Exploratory Data Analysis)

print(df.columns)
 
# Depict count of top 7 companies on a countplot

sns.countplot(y=df['Company'],order=df['Company'].value_counts().sort_values(ascending=False)[:7].index)
plt.title('count of top 7 companies')
plt.show()

# Inference :- Dell,Lenovo and HP are the most sort after brands

# Depict top 6 laptop types based on count

sns.countplot(y=df['TypeName'],order=df['TypeName'].value_counts().sort_values(ascending=False)[:6].index)
plt.title('count of top 6 Laptop types')
plt.show()

# Inference -> Notebook is the most used laptop type

# Depict Top most GPUs Brand used in Laptops

sns.countplot(y=df['Gpu brand'],order=df['Gpu brand'].value_counts().sort_values(ascending=False)[:3].index)
plt.title('count of top GPUs brand used in Laptop')
plt.show()

# inference -> Intel is the most used GPU brand.

# Checking Boxplot for different CPU based on price

sns.boxplot(y=df['Cpu brand'],x=df['Price'])
plt.title('Boxplot for different CPU based on price')
plt.show()

# Checking Boxplot for different Laptop's company based on price

sns.boxplot(y=df['Company'],x=df['Price'])
plt.title('Boxplot for different Laptops company based on price')
plt.show()

# Distribution plot for price

sns.displot(df['Price'])
plt.show()

# Correlation

corr = df.corr()
sns.heatmap(corr,annot=True,cmap='RdBu')
plt.show()

# checking correlation where abs(corr)>0.7

sns.heatmap(corr[abs(corr)>0.7],annot=True,cmap='RdBu')
plt.show()

# Checking For Outliers

df.describe(percentiles=[0.01,0.02,0.03,0.05,0.97,0.98,0.99]).T 
sns.boxplot(x=df['Ram'])
plt.show()

# how many laptops have Ram more than 12 GB

print(df[df['Ram']>12].shape)

# Let's check for more than 20 GB

print(df[df['Ram']>20].shape)

# Now let's check outlier for weight

sns.boxplot(x=df['Weight'])
plt.show()

# Now let's check where weight is more than 3.5 kg 

print(df[df['Weight']>3.5].shape)

# Now let's check outliers for price

sns.boxplot(x=df['Price'])
plt.show()

#Now we check data where price is more than 150000 and 200000

print(df[df['Price']>150000].shape)
print(df[df['Price']>200000].shape) 

# Outlier Handling

df['Weight'] = np.where(df['Weight']>3.5,3.5,df['Weight'])
sns.boxplot(x=df['Weight'])
plt.show()
 
# outliers are removed but still the column are not labelles so, we will handle Encoding categorical columns using column transformer
 
# select x (independent features) and y (dependent features)

x = df.drop('Price',axis=1)
y = df['Price']
print(type(x))
print(type(y)) 
print(x.shape)
print(y.shape)

# split the data set into train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Create funtion to compute MSE,RMSE,MAE,train and test score

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def eval_model(ytest,ypred):
    mae = mean_absolute_error(ytest,ypred)
    mse = mean_squared_error(ytest,ypred) 
    rmse = np.sqrt(mse)
    r2s = r2_score(ytest,ypred)
    print('MAE',mae)
    print('MSE',mse)
    print('RMAE',rmse)
    print('R2 Score',r2s)
    
# Import ML Model building Libraries

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Building Linear Regression Model
 
x_train.dtypes 

# Columns index that needs to undergo OneHotEncoding [0,1,6,9,10]

step1 = ColumnTransformer(transformers=[('ohe',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],remainder='passthrough')

step2 = LinearRegression()
pipe_lr = Pipeline([('step1',step1),('step2',step2)])
pipe_lr.fit(x_train,y_train)
ypred_lr = pipe_lr.predict(x_test)
eval_model(y_test,ypred_lr)

#MAE 13779.8046575224
#MSE 441055854.3552358
#RMAE 21001.329823495365
#R2 Score 0.7029069536619839

 
 # Building Ridge Regression Model
 
step1 = ColumnTransformer(transformers=[('ohe',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],remainder='passthrough')

step2 = Ridge(alpha=2.41)
pipe_rid = Pipeline([('step1',step1),('step2',step2)])
pipe_rid.fit(x_train,y_train)
ypred_rid = pipe_rid.predict(x_test)
eval_model(y_test,ypred_rid)

#MAE 13539.816447226325
#MSE 412006935.28724957
#RMAE 20297.953968004993
#R2 Score 0.7224741621538671


# Building lasso Regression Model

step1 = ColumnTransformer(transformers=[('ohe',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],remainder='passthrough')

step2 = Lasso(alpha=54.56)
pipe_las = Pipeline([('step1',step1),('step2',step2)])
pipe_las.fit(x_train,y_train)
ypred_las = pipe_las.predict(x_test)
eval_model(y_test,ypred_las)

#MAE 13691.56046398258
#MSE 423572538.8049156
#RMAE 20580.877989165467
#R2 Score 0.7146836287149125


# Building Random Forest Model

step1 = ColumnTransformer(transformers=[('ohe',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=80,max_depth=8,min_samples_split=12,random_state=7)
pipe_rf = Pipeline([('step1',step1),('step2',step2)])
pipe_rf.fit(x_train,y_train)
ypred_rf = pipe_rf.predict(x_test)
eval_model(y_test,ypred_rf)

#MAE 11648.137027259972
#MSE 336946176.9617536
#RMAE 18356.093728289623
#R2 Score 0.7730347184443234


# Building Decision Tree Regressor Model

step1 = ColumnTransformer(transformers=[('ohe',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8,min_samples_split=11,random_state=5)
pipe_dt = Pipeline([('step1',step1),('step2',step2)])
pipe_dt.fit(x_train,y_train)
ypred_dt = pipe_dt.predict(x_test)
eval_model(y_test,ypred_dt)

#MAE 12132.196322590713
#MSE 340945360.1042264
#RMAE 18464.70579522529
#R2 Score 0.7703408884204643

# Decision Tree Regressor Model is the best performing model based on RMSE

# Model Saving

import pickle
pickle.dump(pipe_dt,open('dt_model.pkl','wb')) # saving the best performing model
pickle.dump(df,open('data.pkl','wb'))            # saving the dataset

print(df['HDD'].unique())
print(df['SSD'].unique())
print(df['Ram'].unique())

#[   0  500 1000 2000   32  128]
#[ 128    0  256  512   32   64 1000 1024   16  768  180  240    8]
#[ 8 16  4  2 12  6 32 24 64]















 


