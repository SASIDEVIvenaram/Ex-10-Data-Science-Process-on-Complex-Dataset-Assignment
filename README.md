# Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment
# AIM:
To Perform Data Science Process on a complex dataset and save the data to a file.

# ALGORITHM 
## STEP 1 :
Read the given Data 
## STEP 2 :
Clean the Data Set using Data Cleaning Process
## STEP 3 :
Apply Feature Generation/Feature Selection Techniques on the data set 
## STEP 4 :
Apply EDA /Data visualization techniques to all the features of the data set

# CODE:
```
Developed by: SASIDEVI V
Register No: 212222230136
```
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("/content/StudentsPerformance - StudentsPerformance.csv.csv")
print(data)

data.info()

data.isnull().sum()
# data cleaning
data['test preparation course']=data['test preparation course'].fillna(data['test preparation course'].mode()[0])
data['math score']=data['math score'].fillna(data['math score']).fillna(data['math score'].mean())
data['writing score']=data['writing score'].fillna(data['writing score']).fillna(data['reading score'].median())

data.isnull().sum()

data.describe()

data.head()

# removing outliers
Q1=data['math score'].quantile(0.25)
Q3=data['math score'].quantile(0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
df=data[(data['math score']>=lower) & (data['math score']<=upper)] 
print(df)   #new dataframe.


outliers=data[(data['math score']<lower) | (data['math score']>upper)] 
print(outliers)

df.shape

# Feature generation
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
df1=df.copy()
r=['group A','group B','group C','group D','group E']
enc=OrdinalEncoder(categories=[r])
enc.fit_transform(df1[['race/ethnicity']])
df1['neword1']=enc.fit_transform(df1[['race/ethnicity']])
df1 


df2=df1.copy()
le=LabelEncoder()
df2['neword2']=le.fit_transform(df2['race/ethnicity'])
df2

from sklearn.preprocessing import OneHotEncoder
df3=df.copy()
ohe=OneHotEncoder(sparse=False)
enc=pd.DataFrame(ohe.fit_transform(df3[['lunch']]))
df3=pd.concat([df3,enc],axis=1)
df3.head()

!pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df4=df.copy()
newdata=be.fit_transform(df4['test preparation course'])
df4=pd.concat([df,newdata],axis=1)
df4.head()

# heatmap
data.corr()
plt.subplots(figsize=(7,5))
sns.heatmap(data.corr(),annot=True)

# Data visualization
# Scatter plot of math score vs. reading score
plt.scatter(data['math score'], data['reading score'])
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.title('Math Score vs. Reading Score')
plt.show()

sns.barplot(x='gender',y='reading score',data=df)

sns.boxplot(x="math score",data=df)
```
# OUTPUT:
## Dataset
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/c600b2d5-06ba-4d5a-b6d1-d8614af3b58e)
## data.info()
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/0bb8c631-cb0b-41d3-b2ee-8ccd6a9ea3cf)
## data.isnull().sum()
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/f1dbac0d-b825-4bb6-80ca-759e7f872bd5)
## After removing null values
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/3ae0123d-b293-4e7a-9df5-5fc29d935b6a)
## data.describe()
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/7ce49354-dce5-494e-955f-7c6229fe9db8)
## data.head()
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/84c51aa2-bf96-4824-b7f6-0630a21ac1ba)
## New data after removing outliers
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/73995a11-09dc-4f4e-b0cf-760d5dd4db6e)

## Outliers
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/508f49a9-ea3a-4c8f-88d1-af852bf2c664)
## df.shape()
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/9376f475-1a0e-4f01-8aa0-97d6e23c1f69)
## Ordinal Encoding
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/dd6e35a2-fe49-4431-b422-acb95cef21ba)
## Label Encoding
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/57f5dc14-38a1-4b90-ac22-e9bbe56318c3)
## OneHot Encoding
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/edb0bfc1-25b7-44aa-8675-529f8468da1b)
## Binary Encoding
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/29dcae8b-bbf2-4b04-9505-2a604a2c6a36)
## Heatmap
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/1a60c0a0-d334-48f8-8e00-03ee2a9971f5)
## Scatterplot
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/e61dde68-6a99-405e-b9df-81d946b1b14f)
## Barplot
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/423a4445-74c7-4806-9144-c18d0d00af7c)
## Boxplot
![image](https://github.com/SASIDEVIvenaram/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118707332/0c7a26a0-5899-4187-83a8-0d588f757321)

# RESULT:
Hence, Data Science Process is performed on a complex dataset and saved the data to a file.
