import random
import copy


class Neuron:
    matrix = []
    output = 0.0
    correction = 0.0
    bias = 0.0

    def __init__(self, number_of_previous_neurons):
        self.matrix=[]
        for i in range(number_of_previous_neurons):
            self.matrix.append(random.random() * 2 - 1)


class Network:
    web_structure = []
    layers = []

    def __init__(self, web_structure):
        self.web_structure = web_structure

    def Fill(self):
        for i in range(len(self.web_structure)):
            self.layers.append([])
            for j in range(self.web_structure[i]):
                if i == 0:
                    self.layers[i].append(Neuron)
                else:
                    self.layers[i].append(Neuron(self.web_structure[i-1]))


web = Network([5, 2, 1])

web.Fill()

print(web.layers[0][0].matrix)

web.layers[0][0].matrix=[1]

print(web.layers[1][1].matrix)
