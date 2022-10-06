import csv
from tkinter import X

import numpy
import numpy as np


class CSV_Handler:
    def save_bias_weights(network):
        print("saving bias and weights")

        headers = ['layer_id', 'neuron_id', 'bias', 'weights']
        rows = []

        for i in range(0, len(network)):
            for neuron in network[i]:
                rows.append([i, neuron.neuron_id, neuron.bias, neuron.weights])

        with open('bias_weights.csv', 'w', newline='') as f:

            # using csv.writer method from CSV package
            write = csv.writer(f)

            write.writerow(headers)
            write.writerows(rows)

    def load_bias_weights(network):
        print("loading bias and weights")
        import ast
        with open('bias_weights.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')

            # skip first row because it's headers
            next(reader, None)

            for row in reader:
                layer = network[int(row[0])]  # first value is the layer id
                neuron = layer[int(row[1])]  # second value is the neuron id
                neuron.bias = float(row[2])  # third value is the bias
                neuron.weights = ast.literal_eval(row[3])  # fourth value is a string that needs to converted to a list


def load_data(training_size_percent, testing_size_percent):
    print("loading data")

    with open('features_waldo.csv', 'r') as f:
        x = np.loadtxt(f, delimiter=',')
        # add the expected output values as columns to the end of the input values. first column is 1 for waldo, second column is 1 for no waldo

        y = np.ones(shape=(x.shape[0], 1))
        y = np.append(y, np.zeros([len(x), 1]), axis=1)

    

    # with open('features_notwaldo.csv', 'r') as f:
    #    notwaldo = np.loadtxt(f, delimiter=',')
    #    # add the expected output values as columns to the end of the input values. first column is 1 for waldo, second column is 1 for no waldo
    #    notwaldo = np.append(notwaldo, np.zeros([len(notwaldo), 1]), axis=1)
    #    notwaldo = np.append(notwaldo, np.ones([len(notwaldo), 1]), axis=1)

    # data = np.append(data, notwaldo, axis=0)

    # set the random seed to get the same result every run
    np.random.seed(0)

    # get the row count of the matrix
    rows = x.shape[0]

    # shuffle the rows of the matrix
    np.random.shuffle(x)

    # calculate the last row index of the training and testing samples
    last_row_training = int(rows * training_size_percent / 100)
    last_row_testing = last_row_training + int(rows * testing_size_percent / 100)

    # slice the matrix into three by using the row indexes
    x_train = x[:last_row_training]
    y_train = y[:last_row_training]
    x_test = x[last_row_training:last_row_testing]
    y_test = y[last_row_training:last_row_testing]
    x_valid = x[last_row_testing:]
    y_valid = y[last_row_testing:]

    print("sample sizes: data: ", x.shape, " training: ", x_train.shape, " test:", x_test.shape,
          " validation:", x_valid.shape)
    x_train, x_test, y_train, y_test

    return x_train, x_test, x_valid, y_train, y_test, y_valid


class Network:

    def __init__(self, input_layer_size, hidden_layer_count, hidden_layer_size, output_layer_size):

        self.a0 = np.empty(shape=(1, input_layer_size))
        self.a1 = np.empty(shape=(1, hidden_layer_size))
        self.a2 = np.empty(shape=(1, output_layer_size))

        self.activation_layers = []
        self.activation_layers.append(self.a0)
        self.activation_layers.append(self.a1)
        self.activation_layers.append(self.a2)

        self.w1 = np.random.uniform(-1, 1, size=(input_layer_size, hidden_layer_size))
        self.w2 = np.random.uniform(-1, 1, size=(hidden_layer_size, output_layer_size))
        self.b1 = np.random.uniform(-1, 1, size=(1, hidden_layer_size))
        self.b2 = np.random.uniform(-1, 1, size=(1, output_layer_size))
        #self.b0 = np.random.uniform(-1,1, size=(1, input_layer_size))

        #self.w1 = np.ones(shape=(input_layer_size, hidden_layer_size))
        #self.w2 = np.ones(shape=(hidden_layer_size, output_layer_size))
        #self.b1 = np.ones(shape=hidden_layer_size)
        #self.b2 = np.ones(shape=output_layer_size)

        # Gradients
        self.w1_gradient = np.empty(shape=(input_layer_size, hidden_layer_size))
        self.w2_gradient = np.empty(shape=(hidden_layer_size, output_layer_size))
        self.b1_gradients = np.empty(shape=self.b1.shape)
        self.b2_gradients = np.empty(shape=self.b2.shape)
        self.layer_count = 3

        # Z, sum of weighted activation + bias.
        self.z1 = np.empty(hidden_layer_size)
        self.z2 = np.empty(output_layer_size)

    def feed_forward(self, x):
        # populate the input layer with the data_point
        self.a0 = x

        # print("a1.shape: ", self.a1.shape,
        #      " a0.shape: ", self.a0.shape, 
        #      " w1.shape: ", self.w1.shape, 
        #      " b1.shape: ", self.b1.shape)

        # feed forward the input
        relu_v = np.vectorize(lambda x: self.relu(x))
        self.a1 = np.matmul(self.a0, self.w1) + self.b1
        self.z1 = self.a1
        self.a1 = relu_v(self.a1)

        self.a2 = np.matmul(self.a1, self.w2) + self.b2
        self.z2 = self.a2
        self.a2 = relu_v(self.a2)

        # print(self.z2)
        # print(self.a2)

    def backpropagation(self, y):

        # the delta for the current layer is equal to the delta
        # of the *previous layer* dotted with the weight matrix
        # of the current layer, followed by multiplying the delta
        # by the derivative of the nonlinear activation function
        # for the activations of the current layer

        vectorized_d_relu = np.vectorize(lambda x: self.d_relu(x))

        # Partial derivatives for cost with respect to w2, hidden-output gradient.
        dc_a2 = (self.a2 - y) * 2
        da2_z2 = vectorized_d_relu(self.z2)
        dz2_w2 = self.a1
        p = da2_z2 * dc_a2
        # g2 = dc2_w2 = dz2_w2 @ da2_z2 @ dc_a2
        # shape (3,0)   (2,0)    (2,0)

        # Partial derivatives for cost with respect to w1, input-hidden gradient
        dz2_a1 = self.w2
        da1_z1 = vectorized_d_relu(self.z1)
        dz1_w1 = self.a0

        # g1 = c1_w1 = dz1_w1 @ da1_z1 @ dz2_a1 @ p
        #               (2,0)   (3,0)   (3,2)     ?

        # Output - Hidden
        # c_wrt_a2 = (self.a2 - self.get_expected_output(data_point)) * 2
        # a2_wrt_z2 = vectorized_d_relu(self.z2)
        # c_wrt_z2 = a2_wrt_z2 * c_wrt_a2
        # self.w2_gradient += c_wrt_z2 * self.a1[:, np.newaxis]
        # self.b2_gradients += 1 * c_wrt_z2
        # # Output - Hidden => Done!
        #
        # # Hidden - Input
        # c_wrt_a1 = self.w2 @ c_wrt_z2
        # c_wrt_z1 = vectorized_d_relu(c_wrt_a1)
        # self.w1_gradient += c_wrt_z1 * self.a0[:, np.newaxis]
        # self.b1_gradients += 1 * c_wrt_z1
        # # Hidden - Input => Done!

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

    def classify(self):
        print("classification: ", self.a2)

    def cost(self, output, expected_output) -> float:
        error = output - expected_output
        return error * error

    def loss(self, y_1) -> float:
        loss = 0.0
        for i in range(0, self.a2.shape[1]):
            loss += self.cost(self.a2[1][i], y_1[i])

        return loss

    def loss_average(self, y) -> float:
        # print("calculating average loss")

        total_loss = 0.0
        for y_1 in y:
            total_loss += self.loss(y_1)

        return total_loss / y.shape[0]

    def apply_gradient_descent(self, learning_rate, learning_count):
        #weights
        self.w2 = self.w2 - learning_rate * (self.w2_gradient / learning_count)
        self.w1 = self.w1 - learning_rate * (self.w1_gradient / learning_count)

        #biases
        self.b2 = self.b2 - learning_rate * (self.b2_gradients / learning_count)
        self.b1 = self.b1 - learning_rate * (self.b1_gradients / learning_count)

    def reset_gradients(self):
        self.w1_gradient = np.empty(shape=(input_layer_size, hidden_layer_size))
        self.w2_gradient = np.empty(shape=(hidden_layer_size, output_layer_size))
        self.b1_gradients = np.empty(shape=hidden_layer_size)
        self.b2_gradients = np.empty(shape=output_layer_size)

    def learn(self, x, y):
        # Set how many iterations you want to run this training for
        iterations = 1

        # Set your batch size, 100 is a good size
        batch_size = 4
        batch_count = int(x.shape[0] / batch_size)

        # Set your learning rate. 0.1 is a good starting point
        learning_rate = 0.1

        for i in range(0, iterations):
            for j in range(0, batch_count):
                # feed forward the batch
                self.feed_forward(x)

                # calculate gradients for every data point in the batch
                self.backpropagation(y)

                # apply gradient descent to weights and biases using the stored gradients
                self.apply_gradient_descent(learning_rate, batch_size)

                # reset all the stored gradients
                self.reset_gradients()

            print("iteration: ", i, " avg loss: ", self.loss_average(y))


# remainder becomes validation data. sum of batches must not exceed 100%
#x_train, x_test, x_valid, y_train, y_test, y_valid = load_data(training_size_percent=80, testing_size_percent=10)

x_sample = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
y_sample = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

input_layer_size = x_sample.shape[1]
hidden_layer_count = 1
hidden_layer_size = 3
output_layer_size = y_sample.shape[1]
# the training data contains both input values and expected output values
#input_layer_size = len(training_data[0]) - output_layer_size

NN = Network(input_layer_size, hidden_layer_count, hidden_layer_size, output_layer_size)

# CSV_Handler.load_bias_weights(network) # if you're changing the layout of the NN, disable the loading of biases and
# weights for one iteration

# NN.feed_forward(sample_data)

NN.learn(x_sample, y_sample)

# print("average loss for all training data: ", NN.loss_average(training_data))
# print("average loss for all training data: ", NN.loss_average(sample_data))

#NN.classify()

# CSV_Handler.save_bias_weights(network)
