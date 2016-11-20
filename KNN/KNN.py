# KNN.py
import xlrd
import numpy as np
from numpy import *

class KNN(object):
    def loadDataset(self, labelPath, dataPath):
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

    def KnnClassify(self,testX,trainX,labels,K):
        '''
        KNN-Classify, classify one item
        :param testX:   test point
        :param trainX:  train points
        :param labels:  train labels
        :param K:
        :return:        the predict label of train
        '''
        [N,M]=trainX.shape                              # N: data count, M: data dimension

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
            ith_label = labels[sortdiffidx[i]]
            vote[ith_label] = vote.get(ith_label,0)+1   #get(ith_label,0) : if dictionary 'vote' exist key 'ith_label', return vote[ith_label]; else return 0
        sortedvote = sorted(vote.items(),key = lambda x:x[1], reverse = True)
        # 'key = lambda x: x[1]' can be substituted by operator.itemgetter(1)
        return sortedvote[0][0]