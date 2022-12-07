import os
from tkinter import X
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class CSV_Handler:

    def save_bias_weights(network):
        print("saving bias and weights")

        for i in range(len(network.biases)):
            np.savetxt("b{}_hln{}.csv".format(i+1, network.a1.size), network.biases[i], delimiter=",")

        for i in range(len(network.weights)):
            np.savetxt("w{}_hln{}.csv".format(i+1, network.a1.size), network.weights[i], delimiter=",")

    def load_bias_weights(network):
        print("loading bias and weights")

        for i in range(len(network.biases)):
            filename = "b{}_hln{}.csv".format(i+1, network.a1.size)
            if os.path.exists(filename):
                bias = np.loadtxt(filename, delimiter=",", ndmin=2)
                if bias.shape == network.biases[i].shape:
                    print("valid saved bias exists: ", filename)
                    np.copyto(network.biases[i], bias)

        for i in range(len(network.weights)):
            filename = "w{}_hln{}.csv".format(i+1, network.a1.size)
            if os.path.exists(filename):
                weight = np.loadtxt(filename, delimiter=",", ndmin=2)
                if weight.shape == network.weights[i].shape:
                    print("valid saved weights exists: ", filename)
                    np.copyto(network.weights[i], weight)

def load_waldo_data(training_size_percent, testing_size_percent):
    print("loading waldo data")

    with open('features_waldo.csv', 'r') as f:
        x_waldo = np.loadtxt(f, delimiter=',')

        y_waldo = np.ones(shape=(x_waldo.shape[0], 1))
        y_waldo = np.append(y_waldo, np.zeros([len(x_waldo), 1]), axis=1)

    with open('features_notwaldo.csv', 'r') as t:
        x_notwaldo = np.loadtxt(t, delimiter=',')

        y_notwaldo = np.zeros(shape=(x_notwaldo.shape[0], 1))
        y_notwaldo = np.append(y_notwaldo, np.ones([len(x_notwaldo), 1]), axis=1)

    x_mixed = np.append(x_waldo, x_notwaldo, axis=0)
    y_mixed = np.append(y_waldo, y_notwaldo, axis=0)

    # set the random seed to get the same result every run
    np.random.seed(0)

    # get the row count of the matrix
    rows = x_mixed.shape[0]

    # shuffle the rows of the matrix
    randomize = np.arange(len(x_mixed))
    np.random.shuffle(randomize)
    x_mixed = x_mixed[randomize]
    y_mixed = y_mixed[randomize]

    # calculate the last row index of the training and testing samples
    last_row_training = int(rows * training_size_percent / 100)
    last_row_testing = last_row_training + int(rows * testing_size_percent / 100)

    # slice the matrix into three by using the row indexes
    x_train = x_mixed[:last_row_training]
    y_train = y_mixed[:last_row_training]
    x_test = x_mixed[last_row_training:last_row_testing]
    y_test = y_mixed[last_row_training:last_row_testing]
    x_valid = x_mixed[last_row_testing:]
    y_valid = y_mixed[last_row_testing:]

    print("sample sizes: data: ", x_mixed.shape, " training: ", x_train.shape, " test:", x_test.shape,
          " validation:", x_valid.shape)
    x_train, x_test, y_train, y_test

    return x_train, x_test, x_valid, y_train, y_test, y_valid, x_waldo, y_waldo

def load_iris_data():
    print("loading iris data")
    # Iris
    data = load_iris()
    # Dividing the dataset into target variable and features
    X=data.data
    y=data.target
    y_new = np.empty(shape = (X.shape[0], 3))
    for i in range(0, y.shape[0]):
        if y[i] == [0]:
            y_new[i] = [1,0,0]
        elif y[i] == [1]:
            y_new[i] = [0,1,0]
        elif y[i] == [2]:
            y_new[i] = [0,0,1]
    y = y_new

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4)
    return X_train, X_test, y_train, y_test

def load_xor_data():
    print("loading XOR data")
    # XOR training data
    x_sample = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    # XOR expected labels
    y_sample = np.atleast_2d([0, 1, 1, 0]).T

    return x_sample, y_sample

class Network:

    def __init__(self, input_layer_size, hidden_layer_count, hidden_layer_size, output_layer_size, batch_size):

        self.a0 = np.zeros(shape=(batch_size, input_layer_size))
        self.a1 = np.zeros(shape=(batch_size, hidden_layer_size))
        self.a2 = np.zeros(shape=(batch_size, output_layer_size))

        self.activation_layers = [self.a0, self.a1, self.a2]

        self.w1 = np.random.uniform(0.1, 1.0, size=(input_layer_size, hidden_layer_size))
        self.w2 = np.random.uniform(0.1, 1.0, size=(hidden_layer_size, output_layer_size))
        self.b1 = np.random.uniform(0.1, 1.0, size=(batch_size, hidden_layer_size))
        self.b2 = np.random.uniform(0.1, 1.0, size=(batch_size, output_layer_size))

        #self.w1 = np.random.normal(0.5, size=(input_layer_size, hidden_layer_size))
        #self.w2 = np.random.normal(0.5, size=(hidden_layer_size, output_layer_size))
        #self.b1 = np.random.normal(0.5, size=(batch_size, hidden_layer_size))
        #self.b2 = np.random.normal(0.5, size=(batch_size, output_layer_size))

        #self.w1 = np.ones(shape=(input_layer_size, hidden_layer_size))
        #self.w2 = np.ones(shape=(hidden_layer_size, output_layer_size))
        #self.b1 = np.zeros(shape=self.a1.shape)
        #self.b2 = np.zeros(shape=self.a2.shape)

        self.weights = [self.w1, self.w2]
        self.biases = [self.b1, self.b2]

        # Gradients
        self.w1_gradient = np.zeros(shape=self.w1.shape)
        self.w2_gradient = np.zeros(shape=self.w2.shape)
        self.b1_gradients = np.zeros(shape=self.b1.shape)
        self.b2_gradients = np.zeros(shape=self.b2.shape)
 
        # Z, sum of weighted activation + bias.
        self.z1 = np.zeros(hidden_layer_size)
        self.z2 = np.zeros(output_layer_size)

        self.accuracy = 0
        self.true_negative = 0
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def feed_forward(self, x):
        # populate the input layer with the data_point
        self.a0 = x

        # feed forward the input
        relu_v = np.vectorize(lambda x: self.relu(x))
        self.z1 = np.matmul(self.a0, self.w1) + self.b1
        self.a1 = relu_v(self.z1)

        self.z2 = np.matmul(self.a1, self.w2) + self.b2
        self.a2 = relu_v(self.z2)

    def ffw_backprop_gradient_alternative(self, x, y, learning_rate):
        relu_v = np.vectorize(lambda x: self.relu(x))

        # Implementing feedforward propagation on hidden layer
        Z1 = np.dot(x, self.w1)
        self.a1 = relu_v(Z1)
 
        # Implementing feed forward propagation on output layer
        Z2 = np.dot(self.a1, self.w2)
        self.a2 = relu_v(Z2)

        d_relu_v = np.vectorize(lambda x: self.d_relu(x))

        # Backpropagation phase
        E1 = self.a2 - y
        dW1 = E1 * self.a2 * d_relu_v(self.a2)
 
        E2 = np.dot(dW1, self.w2.T)
        dW2 = E2 * self.a1 * d_relu_v(self.a1)
 
        # Updating the weights
        W2_update = np.dot(self.a1.T, dW1) / y.size
        W1_update = np.dot(x.T, dW2) / y.size
 
        self.w2 = self.w2 - learning_rate * W2_update
        self.w1 = self.w1 - learning_rate * W1_update

    def backpropagation(self, y):

        d_relu_v = np.vectorize(lambda x: self.d_relu(x))

        # Partial derivatives - output layer
        dc_a2 = (self.a2 - y) * 2
        da2_z2 = d_relu_v(self.z2)
        dz2_w2 = self.a1
        # Partial derivatives - hidden layer
        dz2_a1 = self.w2
        da1_z1 = d_relu_v(self.z1)
        dz1_w1 = self.a0

        # Matrix multiplication - Cost with respect to weights - output layer
        p = (da2_z2 * dc_a2)
        g2 = dc_w2 = dz2_w2.T @ p   # OK!
        # Matrix multiplication - Cost with respect to weights - hidden layer
        dc_a1 = (p @ dz2_a1.T)   # OK!
        dc_z1 = da1_z1 * dc_a1
        g1 = dc_w1 = dz1_w1.T @ dc_z1

        self.w2_gradient += g2
        self.w1_gradient += g1

        # Matrix multiplication - Cost with respect to bias
        self.b2_gradients += p * 1
        self.b1_gradients += dc_z1 * 1

        # Gradient checking
        # e = 10**0.004
        # f1 = numpy.concatenate((self.w1_gradient.flatten(), self.b1_gradients.flatten()), axis=0) + e
        # f2 = numpy.concatenate((self.w1_gradient.flatten(), self.b1_gradients.flatten()), axis=0) - e
        # theta = numpy.concatenate((self.w1_gradient.flatten(), self.b1_gradients.flatten()), axis=0)
        # dtheta = (f1 - f2)/(2*e)
        # values = (numpy.linalg.norm(dtheta - theta))/(numpy.linalg.norm(dtheta) + numpy.linalg.norm(theta))
        # print(self.w1_gradient)
        # print(values)

    def relu(self, x):
        return max(0.0, x)

    # Derivative of relu, if x > 0, return 1, else 0.
    def d_relu(self, x):
        return 1 * (x > 0)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(self, z):
        return self.sigmoid * (1 - self.sigmoid(z))

    def classify(self, x, y, batch_size):
        batch_count = int(x.shape[0] / batch_size)
        x_batch = np.split(x, batch_count)
        y_batch = np.split(y, batch_count)

        for j in range(0, len(x_batch)):
            # feed forward the batch
            self.feed_forward(x_batch[j])
            #print("classification:\n", self.a2, "\nlabels:\n", y_batch[j])
            self.calculate_accuracy(self.a2, y_batch[j])

        classification_count = int(x_test.shape[0] / batch_size)
        test_waldo_count = np.count_nonzero(y_test[:,1] == 0)
        test_notwaldo_count = np.count_nonzero(y_test[:,1] == 1)

        #NN.classify(x_waldo, y_waldo, batch_size)
        #classification_count = int(x_waldo.shape[0] / batch_size)
        #test_waldo_count = np.count_nonzero(y_waldo[:,1] == 0)
        #test_notwaldo_count = np.count_nonzero(y_waldo[:,1] == 1)

        # prevent division by zero when using lopsided testing samples.
        if test_waldo_count == 0:
            test_waldo_count = 1

        if test_notwaldo_count == 0:
            test_notwaldo_count = 1

        print("Neural Network with", self.a1.size, "hidden layer neurons")
        print("total accuracy: ", (NN.accuracy / classification_count * 100, "%"))
        print("true_positive: ", (NN.true_positive / test_waldo_count * 100, "%"))
        print("true_negative: ", (NN.true_negative / test_notwaldo_count * 100, "%"))
        print("false_positive: ", (NN.false_positive / test_notwaldo_count * 100, "%"))
        print("false_negative: ", (NN.false_negative / test_waldo_count * 100, "%"))


    def calculate_accuracy(self, y_pred, y_true):
        self.true_positive += (y_pred.argmax(axis=1) == 0 and (y_pred.argmax(axis=1) == y_true.argmax(axis=1))).mean() 
        self.true_negative += (y_pred.argmax(axis=1) == 1 and (y_pred.argmax(axis=1) == y_true.argmax(axis=1))).mean()
        self.false_positive += (y_pred.argmax(axis=1) == 0 and (y_pred.argmax(axis=1) != y_true.argmax(axis=1))).mean()
        self.false_negative += (y_pred.argmax(axis=1) == 1 and (y_pred.argmax(axis=1) != y_true.argmax(axis=1))).mean()

        self.accuracy += (y_pred.argmax(axis=1) == y_true.argmax(axis=1)).mean()
        
    def cost(self, output, expected_output):
        error = output - expected_output
        return error**2
     
    def loss(self, y):
        # print("calculating average loss")
        loss = 0.0
        for i in range(0, y.shape[0]): # for every row in expected output
                loss += self.cost(self.a2, y[i]).sum()

        return loss / y.shape[0]

    def apply_gradient_descent(self, learning_rate, learning_count):
        #weights
        #print("_________________")
        #print("w1 grad:", self.w1_gradient)
        #print("w2 grad:", self.w2_gradient)
        np.copyto(self.w2, self.w2 - learning_rate * (self.w2_gradient / learning_count))
        np.copyto(self.w1, self.w1 - learning_rate * (self.w1_gradient / learning_count))

        #biases
        np.copyto(self.b2, self.b2 - learning_rate * (self.b2_gradients / learning_count))
        np.copyto(self.b1, self.b1 - learning_rate * (self.b1_gradients / learning_count))

    def reset_gradients(self):
        self.w1_gradient = np.zeros(shape=self.w1.shape)
        self.w2_gradient = np.zeros(shape=self.w2.shape)
        self.b1_gradients = np.zeros(shape=self.b1.shape)
        self.b2_gradients = np.zeros(shape=self.b2.shape)

    def learn(self, x, y, batch_size, learning_rate, epochs):
        print("training network")
        batch_count = int(x.shape[0] / batch_size)

        for i in range(0, epochs):

            x_batch = np.split(x, batch_count)
            y_batch = np.split(y, batch_count)

            for j in range(0, len(x_batch)):

                #self.ffw_backprop_gradient_alternative(x_batch[j], y_batch[j], learning_rate)

                # feed forward the batch
                self.feed_forward(x_batch[j])

                # calculate gradients for every data point in the batch
                self.backpropagation(y_batch[j])

                # apply gradient descent to weights and biases using the stored gradients
                self.apply_gradient_descent(learning_rate, x_batch[j].size)

                # reset all the stored gradients
                self.reset_gradients()

            print("epoch: ", i, " avg loss: ", self.loss(y))
            CSV_Handler.save_bias_weights(NN)

#waldo data
#remainder becomes validation data. sum of batches must not exceed 100%
x_train, x_test, x_valid, y_train, y_test, y_valid, x_waldo, y_waldo = load_waldo_data(training_size_percent=80, testing_size_percent=20)

##iris data
#x_train, x_test, y_train, y_test = load_iris_data()

##XOR data 
#x_train, y_train = load_xor_data()
#x_test = x_train
#y_test = y_train

print('x_train.shape:', x_train.shape)
print('y_train.shape:', y_train.shape)

input_layer_size = x_train.shape[1]
hidden_layer_count = 1
output_layer_size = y_train.shape[1]
batch_size = 1 # needs to be evenly divideable by training size
learning_rate = 0.1 # Set your learning rate. 0.1 is a good starting point
epochs = 20 # Set how many iterations you want to run the training for

max_hidden_layer_size = 32
hidden_layer_step_size = 4

for i in range(2, max_hidden_layer_size, hidden_layer_step_size):
    print("Creating new network, hidden layer size:", i)
    NN = Network(input_layer_size, hidden_layer_count, i, output_layer_size, batch_size)
    CSV_Handler.load_bias_weights(NN)
    NN.learn(x_train, y_train, batch_size, learning_rate, epochs)
    NN.classify(x_test, y_test, batch_size)

