# scikit_learn
Learning how to use Scikit-learn for a variety of tasks.

Here is a list of each file, and what each does.


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










