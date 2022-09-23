import csv

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
        #network = [list] * (nr_of_input + nr_of_hidden_layers + nr_of_output)

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

    def feed_forward(self):
        for layer in self.network:
            for neuron in layer:
                neuron.output = neuron.bias
                for i in range(0, len(neuron.prev)):
                    neuron.output += neuron.prev[i].output * neuron.weights[i]

                neuron.output = self.relu(neuron.output)

    def print_neurons(self):
        for i in range(0, len(self.network)):
            for neuron in self.network[i]:
                print("layer id: ", i, " neuron id: ", neuron.neuron_id, " inc con:", len(neuron.prev), " bias:", neuron.bias, "weights: ", neuron.weights)

    def save_bias_weights(self):
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
        import ast
        with open('bias_weights.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')

            #skip first row because it's headers
            next(reader, None)

            for row in reader:
                layer = self.network[int(row[0])] #first value is the layer id
                neuron = layer[int(row[1])] #second value is the neuron id
                neuron.bias = int(row[2]) #third value is the bias
                neuron.weights = ast.literal_eval(row[3]) #fourth value is a string that needs to converted to a list

    def cost(self, output, expected_output) -> float:
        error = output - expected_output
        return error * error

    def loss(self, expected_output) -> float:
        self.feed_foward()

        output_layer = self.network.layer[len(self.network) - 1]

        cost = 0.0
        for i in range(0, len(output_layer)):
            cost += self.cost(output_layer[i].output, expected_output[i])

        return cost

    def back_propagation(self, training_data, labels, weights, layer_count):
        dosomething

    def train_network(self):
        #load data
        #split data into train and test data
        #batching
        #run backpropagation
        dosomething

    def classify(self):
        for neuron in self.network[len(self.network) - 1]:
            print("classification, neuron id: ", neuron.neuron_id, " output: " , neuron.output)

NN = Network()
NN.create_network(1, 5, 2, 2)
NN.load_bias_weights() #if changing the layout of the NN disable the loading for one run
NN.feed_forward()
NN.print_neurons()
NN.classify()
NN.save_bias_weights()