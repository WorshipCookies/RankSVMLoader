import itertools
import numpy as np
import sys
import os
import glob

import operator
from scipy import stats
import json
from Tkinter import Tk
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory

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

def calculateAccuracy(rankFileName,globalOrder):
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

def calculateANN(annModel, vectorFile, rankFileName):
    inputLayer = []
    hiddenLayer = []
    outputLayer = []

    globalOrder = dict()

    # LOAD JSON OBJECT
    json_data = open(modelname).read()
    data = json.loads(json_data)

    selectedFeatures = data['selected_features']

    mlpLayers = data['neuralNetworkInfo']['mlpLayers']

    featureVectors = createFeatureDictionary(vectorFile, selectedFeatures)

    # For this particular case we know its a 1 Hidden Layer
    hLayerVal = 0

    if len(mlpLayers) > 2:

        HiddenNeuronList = mlpLayers[1]['layerNeurons']
        OutputNeuronList = mlpLayers[2]['layerNeurons']

        hLayerVal = len(HiddenNeuronList)

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

    return calculateAccuracy(rankFileName,globalOrder), hLayerVal

def writeToFile(dirname, accVector, accMean, layer, maxFoldVal, minFoldVal):
    path = os.path.join(dirname,'testAccExp.csv')
    auxstr = ""

    if not os.path.exists(path):
        auxstr += "Model,Mean,maxFoldVal,minFoldVal,,Fold\n"

    writer = open(path,'a+')

    if layer == 0:
        auxstr += "NoHidden,"
    else:
        auxstr += "Hidden(" + str(layer) + "),"

    auxstr += str(accMean) + "," + str(maxFoldVal) + "," + str(minFoldVal) + ",,"
    for i in accVector:
        auxstr += str(i) + ","
    auxstr+='\n'
    writer.write(auxstr)

# Phases of the ANN Loader

# IMPORTANT STEPS IN ORDER TO USE A PLT ANN MODEL:

# ---------------- LOADING THE MODEL -----------------------
# 1 -- LOAD THE ANN PLT TRAINED MODEL -- JSON OBJECT ( USE THE BEST MODEL OBTAINED! )
# 2 -- LOAD THE DATASET THAT WAS USED (IMPORTANT THE DATASET MUST BE THE SAME USED IN THE MODEL!!!)

# --------------- PREDICTING VALUES ------------------------
# 1 -- USE THE SAME FEATURES OF THE DATASET ( IF SFS OR SBS WAS APPLIED IT WILL AUTOMATICALLY APPLY ON THE FULL DATASET!)
# 2 -- THE OUTPUT WILL BE A GLOBAL RANKING PREDICTION OF ALL THE SOUNDS + THE ONES ADDED

# THIS VALUE CONSISTS OF THE NUMBER OF REPETITIONS
repeat = 5

print 'SELECT FOLDER WITH EXPERIMENTS'

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
dirname = askdirectory(initialdir="D:/GoogleDrive/ParsingExperiment/GeneralExperiment",title='Please select a directory')

print(dirname)

print 'SELECT THE FEATURE DATASET USED FOR TRAINING'

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
vectorFile = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(vectorFile)

# Each Folder is an Experimental Fold
os.chdir(dirname)
foldFiles = [i for i in glob.glob('*.{}'.format('csv'))]
print(foldFiles)

testFoldList = []
foldAverages = []

foldNum = 1

writerOn = True

for fold in foldFiles:
    rankFileName = dirname + '/' + fold
    foldFolderName = fold.split('.')[0]

    print "Calculating Averages of " + foldFolderName

    cPath = dirname + '/' + foldFolderName
    foldPath = os.listdir(cPath)
    expIndex = 0
    for file in foldPath:

        print "Current Index = " + str(expIndex)

        dPath = cPath + '/' + file
        if os.path.isdir(dPath):
            modelDir = os.listdir(dPath)
            fold_num = 0
            acc = 0
            gamAux = 0
            for model in modelDir:
                modelname = dPath + '/' + model
                valAcc, hLayerVal = calculateANN(modelname,vectorFile,rankFileName)
                fold_num+=1
                gamAux = hLayerVal
                acc = valAcc

            if len(foldAverages) <= expIndex:
                foldAverages.append([])

            foldAverages[expIndex].append((acc,gamAux))

            expIndex += 1


runStatVector = []
counter = 0

for f in foldAverages:
    #print f

    if (counter % repeat) == 0:
        runStatVector.append([])

    avg_run = []
    currentLayer = 0
    for run in f:
        avg_run.append(run[0])
        currentLayer = run[1]

    avgFold = np.mean(avg_run)
    foldMax = np.max(avg_run)
    foldMin = np.min(avg_run)

    runStatVector[len(runStatVector)-1].append((avgFold,foldMax,foldMin,currentLayer))

    counter += 1


for st in runStatVector:
    totalMeanVal = []
    maxFoldVal = 0
    minFoldVal = 20000

    numLayer = 0
    for c in st:
        totalMeanVal.append(c[0])
        maxFoldVal = max(maxFoldVal,c[1])
        minFoldVal = min(minFoldVal,c[2])
        numLayer = c[3]

    if numLayer == 0:
        print 'TEST ACCURACY OF NO HIDDEN LAYER = ' + str(np.mean(totalMeanVal))  + ' ; MAX = ' + str(maxFoldVal) + ' ; MIN = ' + str(minFoldVal)
    else:
        print 'TEST ACCURACY OF HIDDEN (' + str(numLayer) + '): MEAN = ' + str(np.mean(totalMeanVal))  + ' ; MAX = ' + str(maxFoldVal) + ' ; MIN = ' + str(minFoldVal)

    if writerOn:
        writeToFile(dirname, totalMeanVal, np.mean(totalMeanVal), numLayer, maxFoldVal, minFoldVal)
