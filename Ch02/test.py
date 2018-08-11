from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
import kNN

group, labels = kNN.createDataSet()
inX = array([0,0])
print kNN.classify0(inX, group, labels, 3)
datingDataMat, datingLabels = kNN.file2matrix('./dataset/datingTestSet2.txt')
print 'datingDataMat[:10]:'
print datingDataMat[:10]
print 'datingLabels[:10]'
print datingLabels[:10]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15*array(datingLabels))
plt.show()

dataset, labels = kNN.file2matrix('./dataset/datingTestSet2.txt')
dataset, ranges, minVals = kNN.autoNorm(dataset)

kNN.datingClassTest()

kNN.classifyPerson()
