#Linear Regression

import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


"""
Boston house prices

Samples total
506

Dimensionality
13

Features
real, positive

Targets
real 5. - 50.
"""



boston = datasets.load_boston()

#features and labels
X, y = boston.data, boston.target

#Can be used to print information about the dataset
# print(X)
# print(X.shape)
# print(y)
# print(y.shape)

#create linear regression algorithm model
l_reg = linear_model.LinearRegression()

#.T reshapes the matrix
# plt.figure()
# plt.scatter(X.T[5], y)
# plt.show()

#Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#train the model
model = l_reg.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions: {}".format(predictions))
print("R^2 values: {}".format(l_reg.score(X, y)))
#slope of all of the fit trendlines
print('coeff: {}'.format(l_reg.coef_))
print('intercept: {}'.format(l_reg.intercept_))

#can now extrapolate and make predictions to other dataspoints using the linear regression model

"""

Classes
2

Samples per class
212(M),357(B)

Samples total
569

Dimensionality
30

Features
real, positive


"""
bc = datasets.load_breast_cancer()
print(bc)

X = scale(bc.data)
y = bc.target
# classes = bc.target_names
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#number clusters => options for y (i.e. in this case 1 or 0, hence 2)
#random_state arbitrary value
model = KMeans(n_clusters=2, random_state=0)

#train, don't need to pass y values, as clusters the data
model.fit(X_train)

predictions = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)

labels = model.labels_


print('labels: {}'.format(labels))
print('predictions: {}'.format(predictions))
print('Accuracy: {}'.format(accuracy))

print(pd.crosstab(y_train, labels))

#are labels wrong?

if ((pd.crosstab(y_train, labels).values[0][0] < pd.crosstab(y_train, labels).values[0][1])
    and (pd.crosstab(y_train, labels).values[1][0] > pd.crosstab(y_train, labels).values[1][1])):
    predictions_relabel = predictions.copy()
    predictions_relabel[np.where(predictions == 0)] = 1
    predictions_relabel[np.where(predictions == 1)] = 0

    accuracy = metrics.accuracy_score(y_test, predictions_relabel)
    print('Accuracy reformatted: {}'.format(accuracy))

