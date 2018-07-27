
# you can change the following hyper-parameter
penalty_factor = 0.0005
learning_rate = [0.001, 0.002, 0.003, 0.004]
max_epoch = 100
sub_samples = 1000
val_sub_samples = 2000
n_features = 123
# ------------------------------------------------------------------

import requests
from sklearn.datasets import load_svmlight_file
from io import BytesIO
import numpy

# load the dataset
def get_dataset(r):
    # load the dataset
    X, y = load_svmlight_file(f=BytesIO(r.content), n_features=n_features)
    # preprocess the dataset
    X = X.toarray()
    n_samples = X.shape[0]
    X = numpy.column_stack((X, numpy.ones((n_samples, 1))))
    y = y.reshape((-1, 1))
    return X, y, n_samples
X_train, y_train, train_samples = get_dataset(requests.get('''https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a'''))
X_val, y_val, val_samples = get_dataset(requests.get('''https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t'''))

# initialize the loss array
losses_val = []

# sample the validation set 
X_val_sub = []
y_val_sub = []
for samples in range(val_sub_samples):
    val_index = numpy.random.randint(0, val_samples - 1)
    X_val_sub.append(X_val[val_index])
    y_val_sub.append(y_val[val_index])

# test different learning rate
for cur_learning_rate in learning_rate:
    losses_val_with_cur_learning_rate = []
    
    # select different initializing method
    w = numpy.zeros((n_features + 1, 1))  # initialize with zeros
    # w = numpy.random.random((n_features + 1, 1))  # initialize with random numbers
    # w = numpy.random.normal(1, 1, size=(n_features + 1, 1))  # initialize with zero normal distribution
    
    # initialize the bias
    b = 0

    # core code of stochastic gradient descent
    for epoch in range(max_epoch):
        X_train_sub = []
        y_train_sub = []
    
        # sample the train set 
        for samples in range(sub_samples):
            train_index = numpy.random.randint(0, train_samples - 1)
            X_train_sub.append(X_train[train_index])
            y_train_sub.append(y_train[train_index])
    
        # I(y_i(w*x_i+b)<1)
        ywx = numpy.dot(numpy.diag(numpy.add(numpy.dot(X_train_sub, w), b).flatten()), y_train_sub)
        indicator = []
        for item in ywx:
            indicator.append(1 if item < 1 else 0)
    
        # calculate the gradient
        G = penalty_factor * w - numpy.divide(numpy.dot(numpy.transpose(X_train_sub), numpy.dot(numpy.diag(indicator), y_train_sub)), sub_samples)
        G = -G
        G_b = numpy.divide(numpy.dot(indicator, y_train_sub), sub_samples)

        # update the parameters
        w += cur_learning_rate * G
        b += cur_learning_rate * G_b

        # predict under the validation set
        ywx = numpy.dot(numpy.diag(numpy.add(numpy.dot(X_train_sub, w), b).flatten()), y_train_sub)
        # calculate the absolute differences
        loss_val = 0
        for item in ywx:
            loss_val += 1 - item if 1 - item > 0 else 0

        loss_val = penalty_factor / 2 * numpy.square(numpy.linalg.norm(w, 2)) + loss_val / val_sub_samples
        losses_val_with_cur_learning_rate.append(loss_val)

    losses_val.append(losses_val_with_cur_learning_rate)

## draw the figure
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))
for i in range(len(learning_rate)):
    plt.plot(losses_val[i], "-", label = "learning rate = " + str(learning_rate[i]))
plt.xlabel("epoch")
plt.ylabel("validation loss")
plt.legend()
plt.title("the graph of absolute diff value varing with the number of iterations.")
plt.show()
## plt.savefig("result.png")
