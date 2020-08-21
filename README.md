# scikit_learn
Learning how to use Scikit-learn for a variety of tasks.

Here is a list detailing what is contained within each file.
Each file is also commented to aid understadning.

<p style='color:red'>This is some red text.</p>
KNN.py:

This is a nearest neighbour classifier.
It uses the classic car evaluation dataset from https://archive.ics.uci.edu/ml/datasets/Car+Evaluation 
This file is saved as car.data in the directory

Each car has seven traits: buying, maint, doors, persons, lug_boot, safety, class.

I seperate the dataset into three features (buying, maintenance and safety) and one target. 

As the dataset is made up of strings, these must be converted to numerical data. 
This is done in two different ways, either with a labelEncoder or manually creating a mapping dictionary.

The data is split into a training sample and a test sample.
The training sample trains the network.
The testing sample tests the accuracy.
Finally the accuracy and an example prection is shown.



SVM.py:

SVM is a classifier algorithm which is effective for smaller datasets.
It creates a hyperplane to seperate the different classes.

The Iris dataset has three classes (the target) and four features (sepal length/width, petal length/width)
The classes are broken into 0, 1 and 2, with the classes list to translate back into strings.

The data is split into a training sample and a test sample.
The training sample trains the network.
The testing sample tests the accuracy.
Finally the accuracy and an example prection is shown.


linear_regression.py

This uses the sklearn module for linear regression.

Firstly it is used on the Boston dataset. 
This has 13 features and 1 target.
The model generates a linear regression between the features and the target, returning coefficients and an intercept  (13 dimensions makes it hard to plot).

The data is split into a training sample and a test sample.
The training sample trains the network.
The testing sample tests the accuracy.
The accuracy, coefficients, R^2 and example prections are printed.

Kmeans.py

Apply a clustering model to the breast cancer dataset from sklearn.
This has 30 features.

This clustering does not require the target values to train
Clustering: This algorithm finds a number of clusters.
This is a grid search, based on moving across the co-oprdinate space, to minimise the distance between dataoints and the clustered centres.



written_number_classifier.py

This uses the sklearn MLPClassifier to classify written numbers.

The process takes the mnist dataset (60000 handwritten numbers) and trains a classifier which can work with other newly created examples.

The mnist dataset is split into a test and trianing set.
Each 28x28 pixel image is flattened and normalised for use in the MLPClassifier.
The classifier is then trained and used to identify the test set of images.

There are two images in this directory, three.png and five.png. 
These were created by me in GIMP, and can be identified using the trained network.







