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

def calculateRBF(x_i, x_j, gamma):
    square_i = calculateDot(x_i,x_i)
    square_j = calculateDot(x_j,x_j)
    return np.exp(-gamma*(square_i+square_j-2*calculateDot(x_i,x_j)))

def calculateLinear(x_i, x_j):
    return calculateDot(x_i,x_j)

def calculateDot(Xi,Xj):
    sum = 0
    for i in range(0,len(Xi)):
        sum += Xi[i] * Xj[i]
    return sum

def runSVM(modelname, vectorFile, rankFileName):
    # print 'LOADING MODEL AND FEATURES INTO MEMORY'

    # LOAD JSON OBJECT
    json_data = open(modelname).read()
    data = json.loads(json_data)

    SV = data['support_vectors']
    alphas = data['alphas']
    selectedFeatures = data['selected_features']

    gamma = 0
    if data['svm_configuration']['kernel_type'] == 'RBF':
        gamma = data['svm_configuration']['gamma']

    # LOAD THE FEATURES OF THE SOUND
    # vectorFile = "dataParamTable(Original)_MFCCAll_Input1_pruned.csv"

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

    # print 'CREATING DATASET ... '

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

    # 2 -- CREATE A STRUCTURE FOR THE SUPPORT VECTORS AND THE ALPHA VALUES
    # Calculate the Support Vector Weights based on either the Linear or RBF functions
    # To calculate the weights we need to get the support vectors of both values in the Support Vector

    # REMINDER IF USING FEATURE SELECTION REMEMBER TO PARSE THE FEATURES OF THE SUPPORT VECTORS SO THAT THEY ARE EQUAL!!!

    # print 'CALCULATING THE KERNELS AND THE GLOBAL ORDER'

    globalOrder = dict()
    if data['svm_configuration']['kernel_type'] == 'RBF':
        for fvect in norm_data:
            SV_Weights = []
            x_i = fvect[1:]
            for vect in SV:
                pref = vect[0]
                other = vect[1]

                wPref = calculateRBF(x_i, featureVectors[pref], gamma)
                wOther = calculateRBF(x_i, featureVectors[other], gamma)
                SV_Weights.append(wPref - wOther)

            orderVal = 0
            for i in range(0, len(SV_Weights)):
                orderVal += alphas[i] * SV_Weights[i]
            globalOrder[fvect[0]] = orderVal

            print "Calculating Support Vectors ..."

    else:
        for fvect in norm_data:
            SV_Weights = []
            x_i = fvect[1:]
            for vect in SV:
                pref = vect[0]
                other = vect[1]

                wPref = calculateLinear(x_i, featureVectors[pref])
                wOther = calculateLinear(x_i, featureVectors[other])
                SV_Weights.append(wPref - wOther)
            orderVal = 0
            for i in range(0, len(SV_Weights)):
                orderVal += alphas[i] * SV_Weights[i]
            globalOrder[fvect[0]] = orderVal

            print "Calculating Support Vectors ..."

    # print 'CALCULATING OBTAINED ACCURACIES'

    # Normalize Values of the Global Rank
    normalizedGlobalOrder = dict()
    maxValue = -sys.float_info.max
    minValue = sys.float_info.max
    for key in globalOrder:
        val = globalOrder[key]
        if (val > maxValue):
            maxValue = val
        if (val < minValue):
            minValue = val

    # for key in globalOrder:
    #     # normalizedGlobalOrder[key] = (2*((globalOrder[key] - minValue) / (maxValue-minValue)))-1
    #     normalizedGlobalOrder[key] = (globalOrder[key] - minValue) / (maxValue-minValue)
    #     print str(key) + " = " + str(normalizedGlobalOrder[key])

    # Save Global Order to File --- DO THIS
    # sorted_x = sorted(normalizedGlobalOrder.items(), key=operator.itemgetter(1), reverse=True)
    # f = open('rankSVM2.csv', 'w')
    # for val in sorted_x:
    #     f.write(str(val[0]) + "," + str(val[1]) + "\n")


    # LOAD FEATURE VECTOR OF NEW SOUNDS TO BE PREDICTED!

    rankFile = open(rankFileName)
    rankData = []
    for line in rankFile:
        if line.strip() != '': # Check if line isn't empty, if it is ignore it.
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

    if gamma == 0:
        print 'TOTAL TEST ACCURACY LINEAR FOLD ' + str(fold) + ' = ' + str(perc)
    else:
        print 'TOTAL TEST ACCURACY OF RBF (' + str(gamma) + ') FOLD ' + str(fold) + ' = ' + str(perc)

    return perc, gamma

def writeToFile(dirname, accVector, accMean, gamma):
    path = os.path.join(dirname,'testAccExp.csv')
    auxstr = ""

    if not os.path.exists(path):
        auxstr += "Model,Mean,,Fold\n"

    writer = open(path,'a+')

    if gamma == 0:
        auxstr += "Linear,"
    else:
        auxstr += "RBF(" + str(gamma) + "),"

    auxstr += str(accMean) + ",,"
    for i in accVector:
        auxstr += str(i) + ","
    auxstr+='\n'
    writer.write(auxstr)

# Phases of the SVMRank Loader

# IMPORTANT STEPS IN ORDER TO USE A PLT MODEL:

# ---------------- LOADING THE MODEL -----------------------
# 1 -- LOAD THE PLT TRAINED MODEL -- JSON OBJECT ( USE THE BEST MODEL OBTAINED! )
# 2 -- LOAD THE DATASET THAT WAS USED (IMPORTANT THE DATASET MUST BE THE SAME USED IN THE MODEL!!!)

# --------------- PREDICTING VALUES ------------------------
# 1 -- USE THE SAME FEATURES OF THE DATASET ( IF SFS WAS APPLIED IT WILL AUTOMATICALLY APPLY ON THE FULL DATASET!)
# 2 -- THE OUTPUT WILL BE A GLOBAL RANKING PREDICTION OF ALL THE SOUNDS + THE ONES ADDED


# SUPPORT VECTORS ARE FLOAT TUPLES, ALPHAS ARE FLOATS AND FEATURE IDENTIFIERS ARE STRINGS
# 1 -- LOAD THE SVM CONFIG, SUPPORT VECTORS, ALPHAS AND FEATURE VECTOR IDENTIFIERS


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
                valAcc, gamma = runSVM(modelname,vectorFile,rankFileName)
                fold_num+=1
                gamAux = gamma
                acc = valAcc

            if len(foldAverages) <= expIndex:
                foldAverages.append([])

            foldAverages[expIndex].append((acc,gamAux))

            expIndex += 1

print foldAverages

for fold in foldAverages:
    avg = []
    gamma = 0
    for exp in fold:
        avg.append(exp[0])
        gamma = exp[1]


    if gamma == 0:
        print 'TEST ACCURACY OF LINEAR: MEAN = ' + str(np.mean(avg))  + ' ; MAX = ' + str(np.max(avg)) + ' ; MIN = ' + str(np.min(avg))
    else:
        print 'TEST ACCURACY OF RBF (' + str(gamma) + '): MEAN = ' + str(np.mean(avg)) + ' ; MAX = ' + str(np.max(avg)) + ' ; MIN = ' + str(np.min(avg))

    if writerOn:
        writeToFile(dirname, avg, np.mean(avg), gamma)


