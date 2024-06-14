# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#---------------------------------------------------------------------

#Data Collection
df=pd.read_csv('C:\\Users\\marin\\Downloads\\diabetes_binary_health_indicators_BRFSS2015.csv')
#---------------------------------------------------------------------

#data cleaning
print("shape before cleaning: ",df.shape)
df.drop_duplicates()
print("Shape after cleaning data set: ",df.shape)
print("null values are \n", df.isnull().sum())
#----------------------------------------------------------------------

#Visualize Correlation
correlation=df.corr()
plt.figure(figsize=(22,22))
g=sns.heatmap(correlation, annot=True, cmap="Pastel1")
plt.show()
#-----------------------------------------------------------------------

#Feature selection
cls=df.corr()['Diabetes_binary'].sort_values().tail(9).index
print(cls)
#-----------------------------------------------------------------------

#Split Data
x=df[cls.drop('Diabetes_binary')]
y=df.Diabetes_binary
#-----------------------------------------------------------------------

#Standard Scale
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
ssx=scaler.transform(x)
#-----------------------------------------------------------------------

#Split data into 80% 20%
x_train,x_test,y_train,y_test=train_test_split(ssx,y,test_size=0.2,random_state=42)
#------------------------------------------------------------------------

#Training By Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='saga',random_state=0)
lr.fit(x_train,y_train)
lr_pre=lr.predict(x_test)
accuracy=lr.score(x_test,y_test)
print("accuracy of logistic regression= ", accuracy * 100, "%")
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, lr_pre)
print(cm)
h=sns.heatmap(cm, annot=True, cmap="RdYlGn",fmt='d')
plt.xlabel("predicted Values")
plt.ylabel("Actual Values")
plt.show()
#---------------------------------------------------------------------------

#Training By SVM Model
from sklearn.svm import LinearSVC
sv=LinearSVC()
sv.fit(x_train,y_train)
svm_pre=sv.predict(x_test)
print(svm_pre.shape)
accuracy = sv.score(x_test, y_test)
print("accuracy of svm= ", accuracy * 100, "%")
cmofsvm=confusion_matrix(y_test,svm_pre)
print(cmofsvm)
h = sns.heatmap(cm,annot=True,cmap="Spectral",fmt='d')
#---------------------------------------------------------------------------

#Training By Decision tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_pre=dt.predict(x_test)
accuracy=dt.score(x_test,y_test)
print("Accuracy of decision tree is: ", accuracy*100, "%" )
cmdt=confusion_matrix(y_test,dt_pre)
print(cmdt)
#--------------------------------------------------------------------------


#Calculate Precision
from sklearn.metrics import classification_report,confusion_matrix
print("Classification Model for Linear Regression")
print(classification_report(y_test,lr_pre,digits=4))
print("Classification Model for SVM")
print(classification_report(y_test,svm_pre,digits=4))
print("Classification Model for Decision Tree")
print(classification_report(y_test,dt_pre,digits=4))
#-------------------------------------------------------------------------

#New Bouns: Combinig Models --> Voting Classifier
from sklearn.ensemble import VotingClassifier
VotingClassifierModel=VotingClassifier(estimators=[('LogisticRegression Model',lr),('SVM Model',sv),('Decision Tree Model',dt)],voting='hard')
VotingClassifierModel.fit(x_train,y_train)
print('Voting Classifier Model Train Score is: ',VotingClassifierModel.score(x_train,y_train)*100,"%")
print('Voting Classifier Model Test Score is: ',VotingClassifierModel.score(x_test,y_test)*100,"%")
#-------------------------------------------------------------------------

#New Bonus: Choosing best hyperparameters
import numpy as np
param_grid = [    
    {'penalty' : ['none'],
    'C' : np.logspace(0,1),
    'solver' : ['saga'],
    'max_iter' : [1000]
    }
]

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(lr, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(x,y)

print(best_clf.best_estimator_)

print (f'Accuracy - : {best_clf.score(x,y)*100}',"%")












