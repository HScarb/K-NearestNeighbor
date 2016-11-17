import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
import os
iris = datasets.load_iris()
iris.data.shape, iris.target.shape
((150, 4), (150,))

def loadDataSet(fileName):
    fr = open(fileName)
    dataMat = []; labelMat = []
    for eachline in fr:
        lineArr = []
        curLine = eachline.strip().split('\t') #remove '\n'
        for i in range(3, len(curLine)-1):
            lineArr.append(float(curLine[i])) #get all feature from inpurfile
        dataMat.append(lineArr)
        labelMat.append(int(curLine[-1])) #last one is class lable
    fr.close()
    return dataMat,labelMat

def splitDataSet(fileName, split_size,outdir):
    if not os.path.exists(outdir): #if not outdir,makrdir
        os.makedirs(outdir)
    fr = open(fileName,'r')#open fileName to read
    num_line = 0
    onefile = fr.readlines()
    num_line = len(onefile)
    arr = np.arange(num_line) #get a seq and set len=numLine
    np.random.shuffle(arr) #generate a random seq from arr
    list_all = arr.tolist()
    each_size = (num_line+1) / split_size #size of each split sets
    split_all = []; each_split = []
    count_num = 0; count_split = 0  #count_num 统计每次遍历的当前个数
                                    #count_split 统计切分次数
    for i in range(len(list_all)): #遍历整个数字序列
        each_split.append(onefile[int(list_all[i])].strip())
        count_num += 1
        if count_num == each_size:
            count_split += 1
            array_ = np.array(each_split)
            np.savetxt(outdir + "/split_" + str(count_split) + '.txt',\
                        array_,fmt="%s", delimiter='\t')  #输出每一份数据
            split_all.append(each_split) #将每一份数据加入到一个list中
            each_split = []
            count_num = 0
    return split_all

def underSample(datafile): #只针对一个数据集的下采样
    dataMat,labelMat = loadDataSet(datafile) #加载数据
    pos_num = 0; pos_indexs = []; neg_indexs = []
    for i in range(len(labelMat)):#统计正负样本的下标
        if labelMat[i] == 1:
            pos_num +=1
            pos_indexs.append(i)
            continue
        neg_indexs.append(i)
    np.random.shuffle(neg_indexs)
    neg_indexs = neg_indexs[0:pos_num]
    fr = open(datafile, 'r')
    onefile = fr.readlines()
    outfile = []
    for i in range(pos_num):
        pos_line = onefile[pos_indexs[i]]
        outfile.append(pos_line)
        neg_line= onefile[neg_indexs[i]]
        outfile.append(neg_line)
    return outfile #输出单个数据集采样结果

def generateDataset(datadir,outdir): #从切分的数据集中，对其中九份抽样汇成一个,\
    #剩余一个做为测试集,将最后的结果按照训练集和测试集输出到outdir中
    if not os.path.exists(outdir): #if not outdir,makrdir
        os.makedirs(outdir)
    listfile = os.listdir(datadir)
    train_all = []; test_all = [];cross_now = 0
    for eachfile1 in listfile:
        train_sets = []; test_sets = [];
        cross_now += 1 #记录当前的交叉次数
        for eachfile2 in listfile:
            if eachfile2 != eachfile1:#对其余九份欠抽样构成训练集
                one_sample = underSample(datadir + '/' + eachfile2)
                for i in range(len(one_sample)):
                    train_sets.append(one_sample[i])
        #将训练集和测试集文件单独保存起来
        with open(outdir +"/test_"+str(cross_now)+".datasets",'w') as fw_test:
            with open(datadir + '/' + eachfile1, 'r') as fr_testsets:
                for each_testline in fr_testsets:
                    test_sets.append(each_testline)
            for oneline_test in test_sets:
                fw_test.write(oneline_test) #输出测试集
            test_all.append(test_sets)#保存训练集
        with open(outdir+"/train_"+str(cross_now)+".datasets",'w') as fw_train:
            for oneline_train in train_sets:
                oneline_train = oneline_train
                fw_train.write(oneline_train)#输出训练集
            train_all.append(train_sets)#保存训练集
    return train_all,test_all