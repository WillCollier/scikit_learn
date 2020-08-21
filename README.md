# scikit_learn
Learning how to use Scikit-learn for a variety of tasks.

Here is a list of each file, and what each does.


KNN.py:

This is a nearest neighbour classifier.
It uses the classic car evaluation dataset from https://archive.ics.uci.edu/ml/datasets/Car+Evaluation 

Each car has seven traits: buying, maint, doors, persons, lug_boot, safety, class.

I seperate the dataset into three features (buying, maintenance and safety) and one target. 

As the dataset is made up of strings, these must be converted to numerical data. 
This is done in two different ways, either with a labelEncoder or manually creating a mapping dictionary.

The data is split into a training sample and a test sample.
The training sample trains the network.
The testing sample tests the accuracy.
Finally the accuracy and an example prection is shown.















