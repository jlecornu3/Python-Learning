# Linear Regression - Boston Data

# Import required libraries
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load Data
boston = datasets.load_boston()

# Split into Features and Labels
X = boston.data
y = boston.target

print(X,y)
print(X.shape)
print(y.shape)

# Instantiate Model Algorithm
lin_reg = linear_model.LinearRegression()

# Visualise the data
# First transpose data to get 13 features
plt.scatter(X.T[5], y)
plt.show()
# Linear relationships so suitable for linear regression

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = lin_reg.fit(X_train,y_train)
predictions = lin_reg.predict(X_test)

print("predictions: ", predictions)
print("R^2: ", lin_reg.score(X,y))
print("coedd:", lin_reg.coef_)
print("intercept:", lin_reg.intercept_)

