import numpy as np

class NeuronLayer():
    def __init__(self, neuron_counts, inputs_per_neuron):
        W = 2 * np.random.random((neuron_counts, inputs_per_neuron)) - 1
        self.weights = np.round (W,3)
        self.bias = np.zeros(neuron_counts)   
            
class NeuralNetwork():    
    def __init__(self, layers = []):      
        self.layers = layers
            
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
              
    def process(self, inputs):
        output_arr = []
        for i in range (0,4):
            s = np.add((self.layers[i].weights).dot(inputs), self.layers[i].bias)
            outputs = self.sigmoid(s)
            output_arr.append(outputs)
            inputs = outputs
        return output_arr
    
    def print_weights(self):
        for i in range (0,4):
            print ("Weights of layer", i+1, "is:")
            print(self.layers[i].weights)

def main():
    np.random.seed(1)
    layer1 = NeuronLayer(5, 5)
    layer2 = NeuronLayer(3, 5)
    layer3 = NeuronLayer(3, 3)
    layer4 = NeuronLayer(2, 3)
    layers = [layer1, layer2, layer3, layer4]

    neural_network = NeuralNetwork(layers)

    print("Beginning Randomly Generated Weights: ")
    neural_network.print_weights()
 
    user_inputs = [float(x) for x in input("Enter user inputs(5): ").split()]

    print("Input data: ", user_inputs)
    print("Output data: ")
    print(neural_network.process(user_inputs))
    
main()