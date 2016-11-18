import numpy as np
import operator

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
    normDataSet = np.zeros(np.shape(dataSet))

    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = dataSet / np.tile(ranges, (m,1))

    return normDataSet, ranges, minVals


def autoNorm2(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = dataSet - np.tile(minVals, (dataSet.shape[0], 1))
    normDataSet = normDataSet / np.tile(ranges, (dataSet.shape[0], 1))
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


def demo(filename, hoRatio, k):
    dataDatingSet, datingLabels = file2matrix(filename)
    normData, ranges, minVals = autoNorm(dataDatingSet)
    m = normData.shape[0]

    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifyResult = classify0(normData[i, :],\
                                   normData[numTestVecs:m, :], \
                                   datingLabels[numTestVecs:m], k)
        if(classifyResult != datingLabels[i]):
            errorCount += 1
            print "the classifier came back with: %d, the real answer is: %d" \
            %(classifyResult, datingLabels[i])
    print "the total error rate is: %f" %(errorCount / float(m))

