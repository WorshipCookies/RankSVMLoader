import itertools
import numpy as np
import sys
import os
import glob
import math

import operator
from scipy import stats
import json
from Tkinter import Tk
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory

inputLayer = []
hiddenLayer = []
outputLayer = []

globalOrder = dict()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))

class neuron:
    def __init__(self, id, inWeight):
        self.id = id
        self.inWeight = inWeight

    def calcOutput(self, inputVal):
        sum = 0
        for i in range(0,len(self.inWeight)):
            w = self.inWeight[i]
            x = inputVal[i]
            sum += w * x

        return sigmoid(sum)

def createFeatureDictionary(vectorFile, selectedFeatures):

    print "Create Feature Dictionary"
    # Obtain the ID position
    featureName = open(vectorFile).next()[:-1].split(',')
    featureID = []
    for f in selectedFeatures:
        index = 0
        for f2 in featureName:
            if f == f2:
                featureID.append(index)
                break
            index += 1

    dataset = np.loadtxt(open(vectorFile, "rb"), delimiter=",", skiprows=1)

    # Normalize DataSet with ZScore
    onlyFeatures = np.delete(dataset, np.s_[:1:1], 1)
    onlyFeatures = stats.zscore(onlyFeatures)

    # Create the final dataset with the selected features
    finalDataSet = np.ndarray(shape=(len(onlyFeatures), len(featureID)))
    for i in range(0, len(onlyFeatures)):
        index = 0
        for j in featureID:
            finalDataSet[i][index] = onlyFeatures[i][(j - 1)]
            index += 1

    idOnly = np.delete(dataset, np.s_[1::1], 1)
    norm_data = np.concatenate((idOnly, finalDataSet), axis=1)  # This is the data points normalized
    # print norm_data

    featureVectors = dict()
    index = 0
    for x in idOnly:
        featureVectors[int(x)] = finalDataSet[index]
        index += 1

    return featureVectors

def calculateANN(annModel, vectorFile):
    # LOAD JSON OBJECT
    json_data = open(modelname).read()
    data = json.loads(json_data)

    selectedFeatures = data['selected_features']

    mlpLayers = data['neuralNetworkInfo']['mlpLayers']

    featureVectors = createFeatureDictionary(vectorFile, selectedFeatures)

    # For this particular case we know its a 1 Hidden Layer

    if len(mlpLayers) > 2:

        HiddenNeuronList = mlpLayers[1]['layerNeurons']
        OutputNeuronList = mlpLayers[2]['layerNeurons']

        # Lets Populate the Input Neurons
        for n in HiddenNeuronList:
            nron = neuron(n['neuronID'], n['weights'])
            hiddenLayer.append(nron)

        for n in OutputNeuronList:
            nron = neuron(n['neuronID'], n['weights'])
            outputLayer.append(nron)

        # Create a predictive global order
        for key in featureVectors.keys():
            featureVect = featureVectors[key].tolist()
            featureVect.insert(0,-1.0) # Add the bias value
            hiddenLayerOutput = []

            for nron in hiddenLayer:
                hiddenLayerOutput.append(nron.calcOutput(featureVect))

            hiddenLayerOutput.insert(0,-1.0) # Add the bias value

            outputLayerOutput = 0
            for nron in outputLayer:
                outputLayerOutput = nron.calcOutput(hiddenLayerOutput)

            globalOrder[key] = outputLayerOutput
    else:
        OutputNeuronList = mlpLayers[1]['layerNeurons']
        for n in OutputNeuronList:
            nron = neuron(n['neuronID'], n['weights'])
            outputLayer.append(nron)

        # Create a predictive global order
        for key in featureVectors.keys():
            featureVect = featureVectors[key].tolist()
            featureVect.insert(0,-1.0) # Add the bias value
            outputLayerOutput = 0

            for nron in outputLayer:
                outputLayerOutput = nron.calcOutput(featureVect)

            globalOrder[key] = outputLayerOutput

            print str(key) + " = " + str(globalOrder[key])

def calculateAccuracy(rankFileName):
    rankFile = open(rankFileName)
    rankData = []
    for line in rankFile:
        if line.strip() != '':  # Check if line isn't empty, if it is ignore it.
            parser = line[:-1]
            parser = parser.split(',')
            rankData.append((int(parser[0]), (int(parser[1]))))

    acc = 0
    for x in rankData:
        pref = x[0]
        other = x[1]

        val_pref = globalOrder[pref]
        val_other = globalOrder[other]

        if val_pref > val_other:
            acc += 1

    perc = acc / float(len(rankData))

    return perc

print 'SELECT THE FEATURE DATASET USED FOR TRAINING'

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
vectorFile = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(vectorFile)

print 'PLEASE CHOOSE THE RANKING FILE FOR TEST ACCURACY'
# Lets check the Accuracy
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
rankFileName = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(rankFileName)

print 'PLEASE CHOOSE THE MODEL FILE TO LOAD'
# Lets check the Accuracy
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
modelname = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(modelname)

calculateANN(modelname,vectorFile)

print "Accuracy = " + str(calculateAccuracy(rankFileName))

