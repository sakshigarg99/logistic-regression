# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:36:36 2020

@author: My
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

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
