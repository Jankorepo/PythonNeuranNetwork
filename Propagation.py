import math
import numpy as np


def goForward(network, inputs_in_list_of_lists):
    inputs = [value for sublist in inputs_in_list_of_lists for value in sublist]
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


def goBackward(network, outputs, learing_rate):
    for layer_number in range(len(network.layers) - 1, -1, -1):
        for neuron_number in range(len(network.layers[layer_number])):
            if layer_number == len(network.layers) - 1:
                network.layers[layer_number][neuron_number].correction = calculateFinalCorr(network,
                                                                                            outputs, layer_number,
                                                                                            neuron_number,
                                                                                            learing_rate)
            else:
                list_of_corrections = getAllCorrOfNextLayer(network.layers[layer_number + 1], neuron_number)
                network.layers[layer_number][neuron_number].correction = calculateHiddenCorr(network,
                                                                                             layer_number,
                                                                                             neuron_number,
                                                                                             list_of_corrections)
    for layer_number in range(1, len(network.layers)):
        for neuron_number in range(len(network.layers[layer_number])):
            upgradeNeuronWeights(network.layers[layer_number][neuron_number], network.layers[layer_number - 1])


def calculateFinalCorr(network, outputs, layer_number, neuron_number, learing_rate):
    return (outputs[neuron_number] - network.layers[layer_number][neuron_number].output) * learing_rate \
           * network.layers[layer_number][neuron_number].output \
           * (1 - network.layers[layer_number][neuron_number].output)


def calculateHiddenCorr(network, layer_number, neuron_number, list_of_corrections):
    return sum(list_of_corrections) * network.layers[layer_number][neuron_number].output \
           * (1 - network.layers[layer_number][neuron_number].output)


def getAllCorrOfNextLayer(next_layer, number_of_neuron):
    tmp_list = []
    for neuron_number in range(len(next_layer)):
        tmp_list.append(next_layer[neuron_number].correction * next_layer[neuron_number].matrix[number_of_neuron + 1])
    return tmp_list


def upgradeNeuronWeights(neuron, previous_layer):
    neuron.matrix[0] = neuron.matrix[0] + neuron.bias * neuron.correction
    prev_layer_outputs = np.array([prev_lay_neuron.output * neuron.correction for prev_lay_neuron in previous_layer])
    neuron.matrix[1:] = neuron.matrix[1:] + prev_layer_outputs
