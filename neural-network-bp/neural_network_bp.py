class Neuron:
    weight = 0
    value = 0
    bias = 0
    prev = []


class Network:
    def __init__(self):
        self.network = []

    def create_layer(self, N) -> list:
        layer = []
        for i in range(0, N):
            layer.append(Neuron)
        return layer

    def connect(self, from_layer, to_layer):
        for i in to_layer:
            for j in from_layer:
                i.prev.append(j)

    def create_network(self, nr_of_hidden_layers, size_of_hidden_layers, nr_of_input, nr_of_output) -> list:
        network = [list] * (nr_of_input + nr_of_hidden_layers + nr_of_output)

        # Create layers
        input_layer = self.create_layer(nr_of_input)
        output_layer = self.create_layer(nr_of_output)
        self.network.append(input_layer)

        for i in range(0, nr_of_hidden_layers):
            hidden_layer = self.create_layer(size_of_hidden_layers)
            self.network.append(hidden_layer)
        self.network.append(output_layer)

        # Connect layers
        for i in range(1, len(self.network)):
            self.connect(self.network[i - 1], self.network[i])
        return self.network

    def feed_forward(self):
        for layer in self.network:
            for node in layer:
                for prev_node in node.prev:
                    node.weight += prev_node.weight
                    node.value += node.value



NN = Network()
NN.create_network(2, 10, 1, 1)
