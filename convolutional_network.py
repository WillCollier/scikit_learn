import numpy as np
import matplotlib.pyplot as plt
import mnist
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


"""
Using the mnist dataset
These are a large sample of hand drawn numbers from 0 to 9

The aim is to use a convolutional neural network to classify the number.

"""

# training variables
x_train_ = mnist.train_images()
y_train = mnist.train_labels()

# testing variables
x_test_ = mnist.test_images()
y_test = mnist.test_labels()

print("x_train: {}".format(x_train_))
print("x_test: {}".format(x_test_))
print("y_train: {}".format(y_train))
print("x_train dimensions: {}".format(x_train_.ndim))
print("x_train shape: {}".format(x_train_.shape))

plt.figure()
plt.imshow(x_train_[7000])
plt.show()

# flatten the array of inputs for use in the neural network
# therefore 28x28 image gets turned into 1 by 28*28
x_train = x_train_.reshape((-1, 28 * 28))
x_test = x_test_.reshape((-1, 28 * 28))

# normalise the inputs to between 0 and 1, as a preferable input format
# Instead of pixel values between 256 and 0
x_train = np.array(x_train) / 256
x_test = np.array(x_test) / 256

# classifier
clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64))

clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
acc = confusion_matrix(y_test, predictions)
print("Accuracy: {}".format(acc))


def accuracy(cm):
    diagonal = cm.trace()
    elements = cm.sum()
    return diagonal / elements


print("Accuracy: {}".format(accuracy(acc)))


"""
Test on some additional hand drawn images, external to the mnist dataset
"""

img = Image.open('five.png')
data = list(img.getdata())
for i in range(len(data)):
    data[i] = 255 - data[i]
# print(data)

five = np.array(data) / 256

p = clf.predict([five])
print(p)

img = Image.open('three.png')
data = list(img.getdata())
for i in range(len(data)):
    data[i] = 255 - data[i]
# print(data)

three = np.array(data) / 256

p = clf.predict([three])
print(p)