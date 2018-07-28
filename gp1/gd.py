import numpy

# you can change the following hyper-parameter
penalty_factor = 0.5  # L2 regular term coefficients
learning_rate = numpy.arange(0.0001, 0.0006, 0.0001)
max_epoch = 200
test_size = 0.25
# ------------------------------------------------------------------

import requests

r = requests.get('''https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale''')

# load the dataset
from sklearn.datasets import load_svmlight_file
from io import BytesIO

X, y = load_svmlight_file(f=BytesIO(r.content), n_features=13)


# preprocess the dataset
X = X.toarray()
n_samples, n_features = X.shape
X = numpy.column_stack((X, numpy.ones((n_samples, 1))))
y = y.reshape((-1, 1))

# devide the dataset into traning set and validation set
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

# initialize the loss array
losses_val = []

for cur_learning_rate in learning_rate:
    cur_losses_val = []
    # select different initializing method
    w = numpy.zeros((n_features + 1, 1))  # initialize with zeros
    # w = numpy.random.random((n_features + 1, 1))  # initialize with random numbers
    # w = numpy.random.normal(1, 1, size=(n_features + 1, 1))  # initialize with zero normal distribution
 
    # core code of gradient descent
    for epoch in range(max_epoch):
        diff = numpy.dot(X_train, w) - y_train
        G = penalty_factor * w + numpy.dot(X_train.transpose(), diff)  # calculate the gradient
        G = -G
        w += cur_learning_rate * G  # update the parameters

        Y_predict = numpy.dot(X_val, w)  # predict under the validation set
        loss_val = numpy.average(numpy.abs(Y_predict - y_val))  # calculate the absolute differences
        cur_losses_val.append(loss_val)

    losses_val.append(cur_losses_val)

# draw the figure
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))
for i in range(len(learning_rate)):
    plt.plot(losses_val[i], "-", label = "learning rate = " + str(learning_rate[i]))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("The graph of absolute diff value varing with the number of iterations.")
plt.show()
# plt.savefig("result.png")
