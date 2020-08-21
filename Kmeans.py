
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

"""
can now extrapolate and make predictions to other data points using the linear regression model
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