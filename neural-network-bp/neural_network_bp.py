class Neuron:
    neuron_id = 0
    output = 0
    bias = 0
    prev = []
    weights = []


class Network:
    def __init__(self):
        self.network = []
        self.neuron_id = 0

    def create_layer(self, N) -> list:
        layer = []
        for i in range(0, N):
            neuron = Neuron()
            neuron.neuron_id = self.neuron_id
            self.neuron_id += 1
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
            for node in layer:
                node.output = node.bias
                for i in range(0, len(node.prev)):
                    node.output += node.prev[i].output * node.weights[i]

                node.output = self.relu(node.output)

    def print_neurons(self):
        for i in range(0, len(self.network)):
            for node in self.network[i]:
                print("layer id: ", i, " node id: ", node.neuron_id, " incoming cons:", len(node.prev))



NN = Network()
NN.create_network(1, 5, 1, 1)
NN.feed_forward()
NN.print_neurons()