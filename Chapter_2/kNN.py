import numpy as np
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                            key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zero(dataSet)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest(filename, hoRatio):

    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        classifferResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], \
                                     datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d"\
        % (classifferResult, datingLabels[i])

        if(classifferResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" %(errorCount / float(numTestVecs))


def classifyPerson(filePath):
    resultList = ['not at all', 'in small does', 'in large does']
    playGameTime = float(raw_input(\
        "percentage of time spent playing video games?"))
    ffMiles = float(raw_input(\
        ""))
    iceCreamCapacity = float(raw_input(\
        "liters of ice cream comsumed per year?"))

    dataSet, labels = file2matrix(filePath)
    normSet, ranges, minVals = autoNorm(dataSet)
    inX = np.array([ffMiles, playGameTime, iceCreamCapacity])
    normInX = (inX - minVals) / ranges
    result = classify0(normInX, normSet, labels)
    print "You will probably like this person: ", \
                resultList[result - 1]


def img2vector(filename):
    returnMat = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnMat[0, 32*i + j] = int(line[j])

    return returnMat


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr =fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' %fileNameStr)
        classifferResult = classify0(vectorUnderTest, \
                                     trainingMat, hwLabels, 3)

        print "the classifier came back with: %d, the real answer is: %d" \
        %(classifferResult, classNumStr)

        if(classifferResult != classNumStr): errorCount += 1.0

    print "\nthe total number iof errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))

