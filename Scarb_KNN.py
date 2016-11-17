from numpy import *
import xlrd     # read xls
from sklearn import cross_validation
from sklearn import
import operator

class KNN:
    def createDataset(self):
        group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
        labels = ['A','A','B','B']
        return group,labels
    def loadDataset(self, labelPath, dataPath):
        labels = [line.strip() for line in open(labelPath)]
        group = []
        workbook = xlrd.open_workbook(dataPath)   # open xls file
        table = workbook.sheets()[0]                    # first sheet
        nrows = table.nrows
        for i in range(nrows):
            print(table.row_values(i))
            group.append(table.row_values(i))
        group = array(group)
        return group,labels

    def KnnClassify(self,testX,trainX,labels,K):
        [N,M]=trainX.shape

    #calculate the distance between testX and other training samples
        difference = tile(testX,(N,1)) - trainX         # tile for array and repeat for matrix in Python, == repmat in Matlab
            # tile: Construct an array by repeating A the number of times given by reps.
        difference = difference ** 2                    # take pow(difference,2)
        distance = difference.sum(1)                    # take the sum of difference from all dimensions
        distance = distance ** 0.5
        sortdiffidx = distance.argsort()

    # find the k nearest neighbours
        vote = {}                                       # create the dictionary
        for i in range(K):
            ith_label = labels[sortdiffidx[i]];
            vote[ith_label] = vote.get(ith_label,0)+1   #get(ith_label,0) : if dictionary 'vote' exist key 'ith_label', return vote[ith_label]; else return 0
        sortedvote = sorted(vote.items(),key = lambda x:x[1], reverse = True)
        # 'key = lambda x: x[1]' can be substituted by operator.itemgetter(1)
        return sortedvote[0][0]

k = KNN() #create KNN object
group, labels = k.loadDataset('dmr_label.txt', 'dmr_data.xls')
# group,labels = k.createDataset()
cls = k.KnnClassify([0,0,0,0,0,0,0,0,0,0,0],group,labels,10)
print(cls)