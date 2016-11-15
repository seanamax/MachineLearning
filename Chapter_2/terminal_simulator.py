import kNN
import numpy
import matplotlib
import matplotlib.pyplot as plt


datingDataSet, datingLabelsSet = kNN.file2matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataSet[:,1],datingDataSet[:,2])
plt.show()
