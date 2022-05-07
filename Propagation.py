import math
import numpy as np


def goForward(network, inputs_in_list_of_lists):
    inputs = np.array([value for sublist in inputs_in_list_of_lists for value in sublist])
    for neuron_number in range(len(network.layers[0])):
        network.layers[0][neuron_number].output = inputs[neuron_number]
    getNeuronsOutputs(network)


def getNeuronsOutputs(network):
    for layer_number in range(1, len(network.layers)):
        for neuron_number in range(len(network.layers[layer_number])):
            neuron_result = calculateSingleNeuronResult(network.layers[layer_number][neuron_number],
                                                        network.layers[layer_number - 1])
            network.layers[layer_number][neuron_number].output = 1 / (1 + (math.exp((-1) * neuron_result)))


def calculateSingleNeuronResult(neuron, previous_layer):
    result = neuron.bias * neuron.matrix[0]
    prev_layer_outputs = np.array([neuron.output for neuron in previous_layer])
    result += sum(prev_layer_outputs * neuron.matrix[1:])
    return result


def goBackward(network, outputs, learning_rate):
    for layer_number in range(len(network.layers) - 1, -1, -1):
        if layer_number == len(network.layers) - 1:
            calcFinalCorrections(network.layers[layer_number], outputs, learning_rate)
        else:
            calcHiddenCorrections(network.layers[layer_number], network.layers[layer_number + 1])

    for layer_number in range(1, len(network.layers)):
        for neuron_number in range(len(network.layers[layer_number])):
            upgradeNeuronWeights(network.layers[layer_number][neuron_number], network.layers[layer_number - 1])


def calcFinalCorrections(neurons, outputs, learning_rate):
    for i in range(len(neurons)):
        neurons[i].correction = (outputs[i] - neurons[i].output) * learning_rate * neurons[i].output * \
                                (1 - neurons[i].output)


def calcHiddenCorrections(layer, next_layer):
    for neuron_number in range(len(layer)):
        sum_of_corrections = 0
        for next_layer_neuron_number in range(len(next_layer)):
            sum_of_corrections += next_layer[next_layer_neuron_number].correction * \
                                  next_layer[next_layer_neuron_number].matrix[neuron_number + 1]
        layer[neuron_number].correction = sum_of_corrections * layer[neuron_number].output * \
                                          (1 - layer[neuron_number].output)


def upgradeNeuronWeights(neuron, previous_layer):
    neuron.matrix[0] = neuron.matrix[0] + neuron.bias * neuron.correction
    prev_layer_outputs = np.array([prev_lay_neuron.output * neuron.correction for prev_lay_neuron in previous_layer])
    neuron.matrix[1:] = neuron.matrix[1:] + prev_layer_outputs
