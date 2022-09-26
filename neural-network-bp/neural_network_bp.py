import csv
import numpy as np


class Neuron:
    def __init__(self):
        self.neuron_id = 0
        self.output = 0
        self.bias = 1
        self.prev = []
        self.weights = []


class Network:
    def __init__(self):
        self.network = []

    def create_layer(self, N) -> list:
        layer = []
        for i in range(0, N):
            neuron = Neuron()
            neuron.neuron_id = i
            layer.append(neuron)
        return layer

    def connect(self, from_layer, to_layer):
        for to_node in to_layer:
            for from_node in from_layer:
                to_node.prev.append(from_node)
                to_node.weights.append(1)

    def create_network(self, nr_of_hidden_layers, size_of_hidden_layers, nr_of_input, nr_of_output) -> list:
        print("creating network")
        # network = [list] * (nr_of_input + nr_of_hidden_layers + nr_of_output)

        # Create layers
        input_layer = self.create_layer(nr_of_input)
        self.network.append(input_layer)

        for i in range(0, nr_of_hidden_layers):
            hidden_layer = self.create_layer(size_of_hidden_layers)
            self.network.append(hidden_layer)

        output_layer = self.create_layer(nr_of_output)
        self.network.append(output_layer)

        # Connect layers
        for i in range(1, len(self.network)):
            self.connect(self.network[i - 1], self.network[i])

        return self.network

    def relu(self, x):
        return max(0.0, x)

    def feed_forward(self, data_point):

        # hook up input layer with the data_point
        for i in range(0, len(data_point) - len(self.network[-1])):
            input_layer = self.network[0]
            input_layer[i].output = data_point[i]

            # calculate output for subsequent layers
        for layer in self.network[1:]:
            for neuron in layer:
                neuron.output = neuron.bias
                for i in range(0, len(neuron.prev)):
                    neuron.output += neuron.prev[i].output * neuron.weights[i]

                neuron.output = self.relu(neuron.output)

    def print_neurons(self):
        print("printing neurons")
        for i in range(0, len(self.network)):
            for neuron in self.network[i]:
                print("layer id: ", i, " neuron id: ", neuron.neuron_id, " inc con:", len(neuron.prev), " bias:",
                      neuron.bias, "weights: ", neuron.weights, "output: ", neuron.output)

    def save_bias_weights(self):
        print("saving bias and weights")

        headers = ['layer_id', 'neuron_id', 'bias', 'weights']
        rows = []

        for i in range(0, len(self.network)):
            for neuron in self.network[i]:
                rows.append([i, neuron.neuron_id, neuron.bias, neuron.weights])

        with open('bias_weights.csv', 'w', newline='') as f:

            # using csv.writer method from CSV package
            write = csv.writer(f)

            write.writerow(headers)
            write.writerows(rows)

    def load_bias_weights(self):
        print("loading bias and weights")
        import ast
        with open('bias_weights.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')

            # skip first row because it's headers
            next(reader, None)

            for row in reader:
                layer = self.network[int(row[0])]  # first value is the layer id
                neuron = layer[int(row[1])]  # second value is the neuron id
                neuron.bias = int(row[2])  # third value is the bias
                neuron.weights = ast.literal_eval(row[3])  # fourth value is a string that needs to converted to a list

    def cost(self, output, expected_output) -> float:
        error = output - expected_output
        return error * error

    def loss(self, data_point) -> float:
        self.feed_forward(data_point)

        output_layer = self.network[-1]

        cost = 0.0
        for i in range(0, len(output_layer)):
            # the expected output are stored at the end of the data_point
            expected_output = data_point[len(data_point) - len(output_layer) + i]
            cost += self.cost(output_layer[i].output, expected_output)

        return cost

    def loss_average(self, data_collection):
        print("calculating average loss")

        total_cost = 0.0
        for data_point in data_collection:
            total_cost += self.loss(data_point)

        return total_cost / len(data_collection)

    def back_propagation(self, layers): # + training data and labels later.

        # Is looking for the negative gradient for the cost function. How far is its current value from its target
        # value, Sigmoid(weighted sum(w1*n1) of all activation in previous layers + bias. Notes: Shuffle training data
        # and apply in minibatches -> Stochastic gradient decent. Less efficient, good approximation. Good
        # computational speed. a(L) = activation_func(w(L)*a(L-1)+b(L) = z(L))

        L = []  # Layers
        w = []  # Weights
        a = []  # Activation values of neurons
        z = []  # A activation_func(sum)
        b = []  # Bias for neurons
        c = []
        y = 0

        # 1d Case
        # C is the result from our backprop.
        c[0] = (a[L] - y)**2
        # Z, activation_func(sum)
        z(L) = w[L] * a[L-1] + b[L]
        a[L] = self.relu(z[L])

        for layer in reversed(layers):
            for neuron in layer:
                for weight in neuron.weights:
                    z = self.relu(weight * layer[layer-1, neuron].output)
                    c = (z - y)**2
                    neuron.output = c
        return 0


    def train_network(self):
        return 0
        # batching
        # run backpropagation
        # dosomething

    def classify(self):
        for neuron in self.network[len(self.network) - 1]:
            print("classification, neuron id: ", neuron.neuron_id, " output: ", neuron.output)


def load_data(training_size_percent, testing_size_percent):
    print("loading data")

    with open('features_waldo.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=',')
        # add the expected output values as columns to the end of the input values. first column is 1 for waldo, second column is 1 for no waldo
        data = np.append(data, np.ones([len(data), 1]), axis=1)
        data = np.append(data, np.zeros([len(data), 1]), axis=1)

    with open('features_notwaldo.csv', 'r') as f:
        notwaldo = np.loadtxt(f, delimiter=',')
        # add the expected output values as columns to the end of the input values. first column is 1 for waldo, second column is 1 for no waldo
        notwaldo = np.append(notwaldo, np.zeros([len(notwaldo), 1]), axis=1)
        notwaldo = np.append(notwaldo, np.ones([len(notwaldo), 1]), axis=1)

    data = np.append(data, notwaldo, axis=0)

    # set the random seed to get the same result every run
    np.random.seed(0)

    # get the row count of the matrix
    rows = data.shape[0]

    # shuffle the rows of the matrix
    np.random.shuffle(data)

    # calculate the last row index of the training and testing samples
    last_row_training = int(rows * training_size_percent / 100)
    last_row_testing = last_row_training + int(rows * testing_size_percent / 100)

    # slice the matrix into three by using the row indexes
    training_data = data[:last_row_training]
    testing_data = data[last_row_training:last_row_testing]
    validation_data = data[last_row_testing:]

    print("sample sizes: data: ", data.shape, " training: ", training_data.shape, " test:", testing_data.shape,
          " validation:", validation_data.shape)

    return training_data, testing_data, validation_data


NN = Network()

training_data, testing_data, validation_data = load_data(training_size_percent=80, testing_size_percent=10)

output_layer_size = 2
# the training data contains both input values and expected output values
input_layer_size = len(training_data[0]) - output_layer_size

NN.create_network(1, 5, input_layer_size, output_layer_size)
NN.load_bias_weights()  # if you're changing the layout of the NN, disable the loading of biases and weights for one iteration

average_loss = NN.loss_average(training_data)
print("average cost: ", average_loss)

NN.print_neurons()
# NN.classify()
NN.save_bias_weights()
