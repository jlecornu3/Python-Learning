# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()

# Split it into features and labels
X = iris.data
y = iris.target

# Check Dimensions
print(X.shape)
print(y.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Check Dimensions
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

# KNN Example
import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')
print(data.head())

X = data[[
    'buying',
    'maint',
    'safety'
]].values

y = data[['class']]
print(X,y)

# Convert String Labels to Codes using LabelEncoder
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

print(X)

# Converting Y using Mapping
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)
print(y)

# Start creating the model - KNN
knn = neighbors.KNeighborsClassifier(n_neighbors=20, weights='uniform')

# Train Model, separate our data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

knn.fit(X_train,y_train)
prediction = knn.predict(X_test)

# Assess accuracy
accuracy = metrics.accuracy_score(y_test, prediction)
print("predictions: ", prediction)
print("accuracy: ", accuracy)

# Check Some Individual Predictions
a = 10
print("actual value: ", y[a])
print("prediction: ", knn.predict(X)[a])

