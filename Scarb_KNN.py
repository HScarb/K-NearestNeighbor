from numpy import *
import numpy as np
import xlrd     # read xls
from sklearn import cross_validation    # for cross_validation iter

def mean_fun(onelist):
    count = 0
    for i in onelist:
        count += i
    return float(count/len(onelist))

def loadDataset(labelPath, dataPath):
    '''
    read data from files
    :param labelPath:   label file path
    :param dataPath     data file path
    :return:            group: data array; labels: label array
    '''
    labels = [line.strip() for line in open(labelPath)]
    group = []
    workbook = xlrd.open_workbook(dataPath)         # open xls file
    table = workbook.sheets()[0]                    # first sheet
    nrows = table.nrows
    for i in range(nrows):
        # print(table.row_values(i))
        group.append(table.row_values(i))
    group = array(group)
    return group,labels

class KNN(object):
    def __init__(self, K):
        self.K = K

    def KnnClassify(self, testItem, trainX, trainY):
        '''
        KNN-Classify, classify one item
        :param testItem:   test point
        :return:           the predict label of train
        '''
        [N,M]=trainX.shape                              # N: data count, M: data dimension

    #calculate the distance between testX and other training samples
        testX2 = tile(testItem, (N, 1))
        difference = tile(testItem, (N, 1)) - trainX         # tile for array and repeat for matrix in Python, == repmat in Matlab
            # tile: Construct an array by repeating A the number of times given by reps.
        difference = difference ** 2                    # take pow(difference,2)
        distance = difference.sum(1)                    # take the sum of difference from all dimensions
        distance = distance ** 0.5
        sortdiffidx = distance.argsort()

    # find the k nearest neighbours
        vote = {}                                       # create the dictionary
        for i in range(self.K):
            ith_label = trainY[sortdiffidx[i]]
            vote[ith_label] = vote.get(ith_label,0)+1   #get(ith_label,0) : if dictionary 'vote' exist key 'ith_label', return vote[ith_label]; else return 0
        sortedvote = sorted(vote.items(),key = lambda x:x[1], reverse = True)
        # 'key = lambda x: x[1]' can be substituted by operator.itemgetter(1)
        return sortedvote[0][0]

    def predict(self, testX, trainX, trainY):
        predY = []
        for item in testX:
            predY.append(self.KnnClassify(item, trainX, trainY))
        return predY

    def accuracy_score(self, trueY, predY):
        score = trueY == predY
        return np.average(score)

if __name__ == '__main__':
    # load data set
    group, labels = loadDataset('dmr_label.txt', 'dmr_data.xls')
    # create KNN object
    k = KNN(K=10)
    # transfer to array-style
    labels = np.array(labels)
    ACCs = []
    # K-fold
    cv = cross_validation.KFold(len(labels), n_folds=10)    # create iterator

    for train_index, test_index in cv:
        x_train, x_test = group[train_index], group[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # generator predict list
        y_pred = k.predict(x_test, x_train, y_train)
        # calculate accuracy
        ACC = k.accuracy_score(y_test, y_pred)
        ACCs.append(ACC)
    ACC_mean = mean_fun(ACCs)
    print('10-fold cross validation accuracy: ', ACC_mean)

    # Leave One Out
    cv = cross_validation.LeaveOneOut(len(labels))      # create iterator

    for train_index, test_index in cv:
        x_train, x_test = group[train_index], group[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        y_pred = k.predict(x_test, x_train, y_train)

        ACC = k.accuracy_score(y_test, y_pred)
        ACCs.append(ACC)
    ACC_mean = mean_fun(ACCs)
    print('Leave-One-Out validation accuracy: ', ACC_mean)

