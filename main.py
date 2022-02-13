import random
import Propagation
from keras.datasets import mnist
import numpy as np


class Neuron:
    def __init__(self):
        self.matrix = []
        self.output = 0.0
        self.correction = 0.0
        self.bias = 1.0

    def setRandomMatrix(self, number_of_previous_neurons):
        self.matrix = np.random.rand(number_of_previous_neurons + 1, 1)
        self.matrix = (self.matrix * 2) - 1
        return self


class Network:
    def __init__(self, web_structure):
        self.web_structure = web_structure
        self.layers = []
        self.fill()

    def fill(self):
        for layer in range(len(self.web_structure)):
            self.layers.append([])
            for number_in_layer in range(self.web_structure[layer]):
                if layer == 0:
                    self.layers[layer].append(Neuron())
                else:
                    self.layers[layer].append(Neuron().setRandomMatrix(self.web_structure[layer - 1]))

    def learnTrainSamples(self, train_samples, answers, epochs):
        for epoch in range(epochs):
            if epoch % 100 == 0:
                print("Próbka numer " + str(epoch))
            line = random.randint(0, len(train_samples) - 1)
            outputs = list(np.zeros(10))
            outputs[answers[line]] = 1
            Propagation.goForward(self, [number / 255 for number in train_samples[line]])
            Propagation.goBackward(self, outputs, 0.1)

    def recognizeTestSamples(self, test_samples, answers):
        sum_of_correct_answers = 0
        for sample_number in range(len(test_samples)):
            if sample_number % 100 == 0:
                print("Test numer " + str(sample_number))
            outputs = list(np.zeros(10))
            outputs[answers[sample_number]] = 1
            Propagation.goForward(self, [number / 255 for number in test_samples[sample_number]])
            result = [neuron.output for neuron in self.layers[-1]]
            if result.index(max(result)) == outputs.index(max(outputs)):
                sum_of_correct_answers += 1
        print("Test samples: " + str(len(test_samples)))
        print("Correct answers: " + str(sum_of_correct_answers))


epoch_number = 10000
web_layers_size = [784, 20, 20, 10]
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

web = Network(web_layers_size)

web.learnTrainSamples(train_X, train_Y, epoch_number)

web.recognizeTestSamples(test_X, test_Y)
