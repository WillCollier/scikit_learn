#KNN classifier

import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


"""
Access the data. 
This is a dataset of cars from a car dealership (https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
The headers of the file is:
buying,maint,doors,persons,lug_boot,safety,class
"""
data = pd.read_csv('car.data')
print(data.head()) #print the top 5 entries in the data file for inspection

"""
Select the rows from the dataset
X are the feature columns 
Y is the target column
"""
X = data[['buying', 'maint', 'safety']].values
y = data[['class']]


#converting strings to numbers w/ LabelEncoder
#This is one option for changing from strings to numbers
#This is because machine learning works with numerical values rather than strings, even if teh numbers are discrete
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])
# print(X)

#converting using a mapping
#Therefore one can manually decide how the strings convert to discrete integers
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)


#create a model
#This sets up the nearest neighbour classifier
#knn object

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
#train the model, requires X, y

#set up the testing and training samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#fit to train the network
knn.fit(X_train, y_train)

#predictions of the network
prediction = knn.predict(X_test)

#find the accuracy
accuracy = metrics.accuracy_score(y_test, prediction)

#Prints all of the predictions and the accuracy of the classifier
print("predictions: {} ".format(prediction))
print("accuracy: {}".format(accuracy))

#Prints out target value a and the actual value
a = 100
print('actual value: {}'.format(y[a]))
print('predicted value: {}'.format(knn.predict(X)[a]))