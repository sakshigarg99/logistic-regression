# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:36:36 2020

@author: My
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

"""df= pd.read_csv('C:/Users/My/Downloads/Heart_disease.csv')

#an= sns.countplot(x='TenYearCHD',hue='age',data=df)
#print(df['age'].plot.hist())iabete
#print(df['education'].plot.hist())

df.dropna(inplace=True)
print(df.isnull().sum())
df.drop(['currentSmoker'],axis=1,inplace=True)
print(df.info())
print(df.head())

plt.scatter(df.age,df['TenYearCHD'],color='r')
plt.show()


X=df.iloc[:,:-1]
Y=df['TenYearCHD']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
reg= LogisticRegression()
reg.fit(X_train,Y_train)
y_pred= reg.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,y_pred))
#print(X_test['age'])

plt.scatter(X_test['age'],y_pred, color='b')
plt.show()
"""
"""df= pd.read_csv('C:/Users/My/Downloads/Social_Network_Ads.csv')
print(df.head())
#a=sns.countplot(x='Purchased',hue='Gender',data=df)
#print(a)
#print(df.isnull().sum())
df.drop(['User ID'],axis=1,inplace=True)
x=df.iloc[:,:-1]
y=df['Purchased']
sex=pd.get_dummies(df['Gender'],drop_first=True)
x=pd.concat([sex,x],axis=1)
x.drop(['Gender'],axis=1,inplace=True)
#print(x.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
reg= LogisticRegression()
reg.fit(x_train,y_train)
y_pred= reg.predict(x_test)

plt.scatter(x_test['EstimatedSalary'],y_pred)
plt.show()

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
"""
df= pd.read_csv('C:/Users/My/Downloads/Social_Network_Ads.csv')
df.drop(['User ID'],axis=1, inplace=True)
print(df.head())
x= df.iloc[:,:-1]
y= df.iloc[:,-1]

sex= pd.get_dummies(df['Gender'], drop_first=True)
x= pd.concat([x,sex], axis=1)
x.drop(['Gender'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
reg= LogisticRegression()
reg.fit(x_train,y_train)

y_pred= reg.predict(sc.transform([[30,87000,1]]))
print(y_pred)

y_pred2= reg.predict(x_test)

#print(np.concatenate((y_test.reshape(len(y_test),1),y_pred2.reshape(len(y_pred2),1)),1))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_pred2))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred2))
