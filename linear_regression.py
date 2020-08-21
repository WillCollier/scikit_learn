#Linear Regression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model



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




