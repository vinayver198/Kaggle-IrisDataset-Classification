# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:03:45 2019

@author: vinayver
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

# Importing the dataset
dataset =  pd.read_csv('Iris.csv')

# Dropping Id column as it will not contribute in the analysis
dataset.drop(['Id'],axis = 1,inplace=True)

dataset.describe()

dataset.info()

#Lets visualize the count of each species
sns.countplot(dataset.Species)

# Let's Visualize the pair plot
g = sns.pairplot(dataset, hue='Species')
plt.show()
# Observation: PetalLength and PetalWidth seems to be greatly affecting the classes


#Creating target variable
Y = dataset['Species']
X = dataset.iloc[:,2:4]

labelEncoder = LabelEncoder()
Y_encoded    = labelEncoder.fit_transform(Y)

# Splitting the dataset
X_train,X_test,Y_train,Y_test = train_test_split(X,Y_encoded,random_state = 4,test_size = 0.2)

#Scaling the features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)

# Let's test this problem statement with different classifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logit_model = LogisticRegression()
logit_model.fit(X_train,Y_train)
logit_pred = logit_model.predict(X_test)
logit_accuracy = metrics.accuracy_score(Y_test,logit_pred)
print("Accuracy with Logistic Regression is {}".format(logit_accuracy))

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train,Y_train)
tree_pred = tree_model.predict(X_test)
tree_accuracy = metrics.accuracy_score(Y_test,tree_pred)
print("Accuracy with Decision Tree is {}".format(tree_accuracy))

# Support Vector Machine
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train,Y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = metrics.accuracy_score(Y_test,svm_pred)
print("Accuracy with SVM is {}".format(svm_accuracy))