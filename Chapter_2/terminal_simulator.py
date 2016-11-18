import kNN
import numpy
import matplotlib
import matplotlib.pyplot as plt
import datetime


# datingDataSet, datingLabelsSet = kNN.file2matrix('datingTestSet2.txt')
# # fig = plt.figure()
# # ax1 = fig.add_subplot(111)
# # ax1.scatter(datingDataSet[:,1],datingDataSet[:,2])
# #
# # plt.show()
#

startTime = datetime.datetime.now()
kNN.demo('datingTestSet2.txt', 0.5, 3)
endTime = datetime.datetime.now()

print endTime - startTime

