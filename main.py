import random
import Propagation
from keras.datasets import mnist


class Neuron:
    matrix = []
    output = 0.0
    correction = 0.0
    bias = 1.0

    def __init__(self):
        self.matrix = []

    def SetRandomMatrix(self, number_of_previous_neurons):
        self.matrix = []
        for i in range(number_of_previous_neurons + 1):
            self.matrix.append(random.random() * 2 - 1)
        return self


class Network:
    def __init__(self, web_structure):
        self.web_structure = web_structure
        self.layers = []

    def Fill(self):
        for i in range(len(self.web_structure)):
            self.layers.append([])
            for j in range(self.web_structure[i]):
                if i == 0:
                    self.layers[i].append(Neuron())
                else:
                    self.layers[i].append(Neuron().SetRandomMatrix(self.web_structure[i - 1]))


(train_X, train_y), (test_X, test_y) = mnist.load_data()
web = Network([784, 20, 20, 10])
web.Fill()
last_line = -1
for i in range(50000):
    if i % 1000 == 0:
        print(i)
    line = random.randint(0, 59999)
    last_line = line
    output = train_X[line]
    outputs = []
    for j in range(len(web.layers[-1])):
        if j == train_y[line]:
            outputs.append(1)
        else:
            outputs.append(0)
    Propagation.goForward(web, [number / 255 for number in train_X[line]])
    Propagation.goBackward(web, outputs, 0.1)

for i in web.layers[-1]:
    print(i.output)

print(train_y[last_line])
