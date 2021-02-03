# ML in SciKit Learn Examples

from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score

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

# Support Vector Machines
iris_svm = datasets.load_iris()

# Split it into features and labels
X_svm = iris_svm.data
y_svm = iris_svm.target

classes = ['Iris Setosa', 'Iris Versicolour','Iris Virginica']
print(X_svm.shape)
print(y_svm.shape)

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm,y_svm,test_size=0.2)

model_svm = svm.SVC()
model_svm.fit(X_train_svm,y_train_svm)
print(model_svm)

prediction_svm = model_svm.predict(X_test_svm)
acc_svm = accuracy_score(y_test_svm, prediction_svm)
print("SVM predictions: ", prediction_svm)
print("SVM actual:", y_test_svm)
print("SVM accuracy: ", acc_svm)

# Look at Names in ForLoop
for i in range(len(prediction_svm)):
    print(classes[prediction_svm[i]])
