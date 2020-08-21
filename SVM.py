#SVM -- Support Vector Machine
#SVM example
#Given input data, identify the iris
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm

"""
The data is imported from sklearn.
This is the irris dataset, which describes three different types of Iris
The features are sepal length and width, petal length and width
The target is the class, either Iris Setosa, Versicolour or Virginica
"""

iris = datasets.load_iris()
#split into features and labels
X = iris.data
y = iris.target
classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

#Uncomment these to print the shape of the features and target
# print(X.shape)
# print(y.shape)

#hours of study vs good/bad grades
#10 different students
#train with 8
#predict with remaining 2
#test level of accuracy

#test_size = fraction of data to be used as testing sample 0.2=20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#Generate the model and train it
model = svm.SVC()
model.fit(X_train, y_train)

# Make the predictions
predictions = model.predict(X_test)

#Test the accuracy
accuracy = metrics.accuracy_score(y_test, predictions)


print("predictions: {} ".format(predictions))
print("actual {}".format(y_test))
print("accuracy: {}".format(accuracy))

# Print each prediction
for i, prediction in enumerate(predictions):
    print(classes[prediction])