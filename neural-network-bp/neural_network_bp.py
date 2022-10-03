import csv
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
        data = np.loadtxt(f, delimiter=',')
        # add the expected output values as columns to the end of the input values. first column is 1 for waldo, second column is 1 for no waldo
        data = np.append(data, np.ones([len(data), 1]), axis=1)
        data = np.append(data, np.zeros([len(data), 1]), axis=1)

    # with open('features_notwaldo.csv', 'r') as f:
    #    notwaldo = np.loadtxt(f, delimiter=',')
    #    # add the expected output values as columns to the end of the input values. first column is 1 for waldo, second column is 1 for no waldo
    #    notwaldo = np.append(notwaldo, np.zeros([len(notwaldo), 1]), axis=1)
    #    notwaldo = np.append(notwaldo, np.ones([len(notwaldo), 1]), axis=1)

    # data = np.append(data, notwaldo, axis=0)

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


class Network:

    def __init__(self, input_layer_size, hidden_layer_count, hidden_layer_size, output_layer_size):
        self.input_layer_activation = np.empty(input_layer_size)
        self.hidden_layer_activation = np.empty(hidden_layer_size)
        self.output_layer_activation = np.empty(output_layer_size)

        self.input_hidden_weights = np.random.uniform(-1, 1, size=(input_layer_size, hidden_layer_size))
        self.hidden_output_weights = np.random.uniform(-1, 1, size=(hidden_layer_size, output_layer_size))

        # self.input_layer_bias = np.random.uniform(-1,1, size=(1, input_layer_size))
        self.hidden_layer_bias = np.random.uniform(-1, 1, size=(hidden_layer_size))
        self.output_layer_bias = np.random.uniform(-1, 1, size=(output_layer_size))

        # Gradients
        self.input_hidden_gradient = np.empty(shape=(input_layer_size, hidden_layer_size))

        self.hidden_output_gradient = np.empty(shape=(hidden_layer_size, output_layer_size))
        self.hidden_layer_gradient = np.empty(shape=hidden_layer_size)

        self.output_layer_gradient = np.empty(shape=(output_layer_size))
        self.layer_count = input_layer_size + hidden_layer_count + output_layer_size

    def feed_forward(self, data_point):
        # populate the input layer with the data_point
        for i in range(len(data_point) - len(self.output_layer_activation)):
            self.input_layer_activation[i] = data_point[i]

        # print("hidden_layer_activation.shape: ", self.hidden_layer_activation.shape,
        #      " input_layer_activation.shape: ", self.input_layer_activation.shape, 
        #      " input_hidden_weights.shape: ", self.input_hidden_weights.shape, 
        #      " hidden_layer_bias.shape: ", self.hidden_layer_bias.shape)

        # feed forward the input
        self.hidden_layer_activation = np.matmul(self.input_layer_activation,
                                                 self.input_hidden_weights) + self.hidden_layer_bias
        self.output_layer_activation = np.matmul(self.hidden_layer_activation,
                                                 self.hidden_output_weights) + self.output_layer_bias

    def backpropagation(self, data_point):

        for layer in range(self.layer_count - 1):
            z_output = np.empty(1, 2)

            f = lambda x: self.d_relu(x)
            self.output_layer_gradient = np.vectorize(f, self.output_layer_activation)
            print(self.output_layer_gradient.shape)

    def relu(self, x):
        return max(0.0, x)

    # Derivative of relu, if x > 0, return 1, else 0.
    def d_relu(self, x):
        return 1 * (x > 0)

    def cost(self, output, expected_output) -> float:
        error = output - expected_output
        return error * error

    def get_expected_output(self, data_point, output_layer, i):
        return data_point[len(data_point) - len(output_layer) + i]

    def classify(self):
        print("classification: ", self.output_layer_activation)

    def loss(self, data_point):
        loss = 0.0
        for i in range(0, len(self.output_layer_activation)):
            # the expected output are stored at the end of the data_point
            expected_output = self.get_expected_output(data_point, self.output_layer_activation, i)
            loss += self.cost(self.output_layer_activation[i], expected_output)

        return loss

    def loss_average(self, data_collection):
        print("calculating average loss")

        total_loss = 0.0
        for data_point in data_collection:
            total_loss += self.loss(data_point)

        return total_loss / len(data_collection)

    def learn(self, training_data):
        # Set how many iterations you want to run this training for
        iterations = 5

        # Set your batch size, 100 is a good size
        batch_size = 1
        batch_count = int(training_data.shape[0] / batch_size)

        # Set your learning rate. 0.1 is a good starting point
        learning_rate = 0.1

        for i in range(0, iterations):
            batch_index = 0
            batch_index_cap = batch_size

            for j in range(0, batch_count):

                for k in range(batch_index, batch_index_cap):
                    # load data point
                    data_point = training_data[k]

                    # feed forward the data point
                    self.feed_forward(data_point)

                    # calculate gradients for every data point in the batch
                    self.backpropagation(data_point,)
                    # self.calculate_gradients(data_point)

                # apply gradient descent to every neuron based on the stored sensitivies
                # self.apply_all_gradient_descent(learning_rate, batch_size)

                # reset all the stored gradients
                # self.reset_gradients()

                # run next batch
                batch_index += batch_size
                batch_index_cap += batch_size

           # print("iteration: ", i, " avg loss: ", self.loss_average(training_data))


# region old

class Neuron:
    def __init__(self):
        self.neuron_id = 0
        self.prev = []
        self.weighted_input = 0.0
        self.output = 0
        self.bias = 1.0
        self.bias_gradient = 0.0
        self.weights = []
        self.weights_gradient = []
        self.weighted_input_deriv = 0.0

    def reset_sensitivies(self):
        self.output_gradient = 0.0
        self.bias_gradient = 0.0
        for x in self.weights_gradient:
            x = 0.0

    # applies gradient descent for the stored incoming weights and bias for this neuron
    def apply_gradient_descent(self, learning_rate, learning_count):
        for i in range(0, len(self.weights)):
            self.weights[i] += learning_rate * -(self.weights_gradient[i] / learning_count)

        self.bias += learning_rate * -(self.bias_gradient / learning_count)


def print_neurons(self):
    print("printing neurons")
    for i in range(0, len(self.network)):
        for neuron in self.network[i]:
            print("layer id: ", i, " neuron id: ", neuron.neuron_id, " inc con:", len(neuron.prev), " bias:",
                  neuron.bias, "weights: ", neuron.weights, "output: ", neuron.output)


def apply_all_gradient_descent(self, learning_rate, learning_count):
    for layer in self.network:
        for neuron in layer:
            neuron.apply_gradient_descent(learning_rate, learning_count)


def reset_gradients(self):
    for layer in self.network[1:]:
        for neuron in layer:
            neuron.reset_sensitivies()


def cost_wrt_activation_derivative(self, output, expected_output):
    return 2 * (output - expected_output)


def activation_wrt_weighted_input_derivative(self, weighted_input):
    return self.d_relu(weighted_input)


def calculate_output_layer_gradients(self, data_point):
    # calculate output layer values
    output_layer = self.network[-1]
    for i in range(0, len(output_layer)):
        neuron = output_layer[i]

        # calculate how the activation affects the cost
        cost_deriv = self.cost_wrt_activation_derivative(neuron.output,
                                                         self.get_expected_output(data_point, output_layer, i))

        # calculate how the weighted input affects the activation
        activation_deriv = self.activation_wrt_weighted_input_derivative(neuron.weighted_input)

        # calculate how the weighted input affects the cost
        neuron.weighted_input_deriv = cost_deriv * activation_deriv

        # calculate how the weights affects the cost
        for j in range(0, len(neuron.weights)):
            neuron.weights_gradient[j] += neuron.weights[j] * neuron.weighted_input_deriv

        # calculate how the bias affects the cost
        neuron.bias_gradient += 1 * neuron.weighted_input_deriv

    return output_layer


def calculate_hidden_layer_gradients(self, hidden_layer, prev_layer):
    for i in range(0, len(hidden_layer)):
        neuron = hidden_layer[i]

        # calculate how the weight affects the cost of the previous layer
        for old_neuron in prev_layer:
            neuron.weighted_input_deriv += old_neuron.weights[i] * old_neuron.weighted_input_deriv

        # calculate how the weighted input affects the activation
        activation_deriv = self.activation_wrt_weighted_input_derivative(neuron.weighted_input)

        neuron.weighted_input_deriv *= activation_deriv

        # calculate how the weights affects the cost
        for j in range(0, len(neuron.weights)):
            neuron.weights_gradient[j] += neuron.weights[j] * neuron.weighted_input_deriv

        # calculate how the bias affects the cost
        neuron.bias_gradient += 1 * neuron.weighted_input_deriv

    return hidden_layer


def calculate_gradients(self, data_point):
    prev_layer = self.calculate_output_layer_gradients(data_point)

    for layer in reversed(self.network[1:-1]):
        prev_layer = self.calculate_hidden_layer_gradients(layer, prev_layer)

    # y = 0
    # l = 0
    # for layer in reversed(self.network):
    #    n = 0
    #    l += 1
    #    for neuron in layer:
    #        # Compute relevant derivatives
    #        # derivative of  C with respect to current neuron activation.
    #        # 2*(a(L)-y)
    #        #dC = 2(neuron.output - y)

    #        # derivative of current neuron activation with respect to sum of previous layer, i.e. w(L)*a(L-1)+b(L)
    #        # dReLU(z(L))
    #        #dA = self.d_relu(z)

    #        # dz(L) with respect to w(L)
    #        # a(L-1)
    #        return 0


# endregion

# training_data, testing_data, validation_data = load_data(training_size_percent=80, testing_size_percent=10)
sample_data = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])

# the training data contains both input values and expected output values
# input_layer_size = len(training_data[0]) - output_layer_size
input_layer_size = 2
hidden_layer_count = 1
hidden_layer_size = 3
output_layer_size = 2

NN = Network(input_layer_size, hidden_layer_count, hidden_layer_size, output_layer_size)

# CSV_Handler.load_bias_weights(network) # if you're changing the layout of the NN, disable the loading of biases and
# weights for one iteration

# NN.feed_forward(sample_data)

NN.learn(sample_data)

# print("average loss for all training data: ", NN.loss_average(training_data))
print("average loss for all training data: ", NN.loss_average(sample_data))

# NN.print_neurons()
NN.classify()

# CSV_Handler.save_bias_weights(network)
