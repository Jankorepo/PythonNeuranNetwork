import math


def goForward(network, inputs):
    for i in range(len(network.layers[0])):
        network.layers[0][i].output = inputs[i]
    getNeuronsOutputs(network)


def getNeuronsOutputs(network):
    for i in range(1, len(network.layers)):
        for j in range(len(network.layers[i])):
            neuron_result = calculateSingleNeuronResult(network.layers[i][j], network.layers[i - 1])
            network.layers[i][j].output = 1 / (1 + (math.exp((-1) * neuron_result)))


def calculateSingleNeuronResult(neuron, previous_layer):
    result = neuron.bias * neuron.matrix[0]
    for i in range(len(previous_layer)):
        result = result + previous_layer[i].output * neuron.matrix[i + 1]
    return result


def goBackward(network, outputs, learing_rate):
    for i in range(len(network.layers)-1, -1, -1):
        for j in range(len(network.layers[i])):
            if i == len(network.layers) - 1:
                network.layers[i][j].correction = (outputs[j] - network.layers[i][j].output) \
                                                  * learing_rate * network.layers[i][j].output \
                                                  * (1 - network.layers[i][j].output)
            else:
                list_of_corrections = getAllCorrectionsOfNextLayer(network.layers[i + 1], j)
                network.layers[i][j].correction = getCorrectionOfNeuronInHiddenLayer(list_of_corrections) \
                                                  * network.layers[i][j].output * (1 - network.layers[i][j].output)
    for i in range(1, len(network.layers)):
        for j in range(len(network.layers[i])):
            upgradeNeuronWeights(network.layers[i][j], network.layers[i - 1])


def getAllCorrectionsOfNextLayer(next_layer, number_of_neuron):
    tmp_list = []
    for i in range(len(next_layer)):
        tmp_list.append(next_layer[i].correction * next_layer[i].matrix[number_of_neuron + 1])
    return tmp_list


def getCorrectionOfNeuronInHiddenLayer(list_of_corrections):
    sum_of_corrections = 0
    for i in list_of_corrections:
        sum_of_corrections = sum_of_corrections + i
    return sum_of_corrections


def upgradeNeuronWeights(neuron, previous_layer):
    neuron.matrix[0] = neuron.matrix[0] + neuron.bias * neuron.correction
    for i in range(len(neuron.matrix)):
        neuron.matrix[i] += previous_layer[i - 1].output * neuron.correction
