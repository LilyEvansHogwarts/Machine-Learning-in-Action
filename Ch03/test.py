import trees

myDat, labels = trees.createDataSet()
print trees.calcShannonEnt(myDat)

print trees.chooseBestFeatureToSplit(myDat)

tree = trees.createTree(myDat, labels)
print tree

def retrieveTree(i):
    listOfTrees = [{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
            {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
    return listOfTrees[i]

print retrieveTree(0)
print retrieveTree(1)

print trees.getNumLeafs(retrieveTree(1))
print trees.getTreeDepth(retrieveTree(1))
