def loadDataSet(fileName):
    dataSet = []
    labels = []
    fileIn = open(fileName)
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])
        labels.append(float(lineArr[2]))
    dataSet = mat(dataSet)
    labels = mat(labels).T
    return dataSet , labels
def selectJrand(i,m):
    j = i
    while(i == j):
        j = int(random.uniform(0,m))
    return j
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj
