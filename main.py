import random
import copy
import Propagation


class Neuron:
    matrix = []
    output = 0.0
    correction = 0.0
    bias = 0.0

    def __init__(self, number_of_previous_neurons):
        self.matrix = []
        for i in range(number_of_previous_neurons+1):
            self.matrix.append(random.random() * 2 - 1)


class Network:
    def __init__(self, web_structure):
        self.web_structure = web_structure
        self.layers = []

    def Fill(self):
        for i in range(len(self.web_structure)):
            self.layers.append([])
            for j in range(self.web_structure[i]):
                if i == 0:
                    self.layers[i].append(Neuron)
                else:
                    self.layers[i].append(Neuron(self.web_structure[i - 1]))


web = Network([2, 2, 1])
web.Fill()

for i in range(100000):
    Propagation.goForward(web, [0, 1])
    Propagation.goBackward(web, [1], 0.01)

print(web.layers[2][0].output)
