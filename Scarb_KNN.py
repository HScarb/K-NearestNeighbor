from numpy import *
import numpy as np
import xlrd     # read xls
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import operator

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

from own_cross_validation import performance


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
            ith_label = labels[sortdiffidx[i]]
            vote[ith_label] = vote.get(ith_label,0)+1   #get(ith_label,0) : if dictionary 'vote' exist key 'ith_label', return vote[ith_label]; else return 0
        sortedvote = sorted(vote.items(),key = lambda x:x[1], reverse = True)
        # 'key = lambda x: x[1]' can be substituted by operator.itemgetter(1)
        return sortedvote[0][0]

k = KNN() #create KNN object
group, labels = k.loadDataset('dmr_label.txt', 'dmr_data.xls')
labels = np.array(labels)
# group,labels = k.createDataset()
cls = k.KnnClassify([0 for i in range(11)],group,labels,10)
print(cls)

#=======Cross validation========
def classifier(clf,train_X, train_y, test_X, test_y):#X:训练特征，y:训练标号
    # train with randomForest
    print(" training begin...")
    clf = clf.fit(train_X,train_y)
    print(" training end.")
    #==========================================================================
    # test randomForestClassifier with testsets
    print(" test begin.")
    predict_ = clf.predict(test_X) #return type is float64
    proba = clf.predict_proba(test_X) #return type is float64
    score_ = clf.score(test_X,test_y)
    print(" test end.")
    #==========================================================================
    # Modeal Evaluation
    ACC = accuracy_score(test_y, predict_)
    SN,SP = performance(test_y, predict_)
    MCC = matthews_corrcoef(test_y, predict_)
    #AUC = roc_auc_score(test_labelMat, proba)
    #==========================================================================
    #save output
    eval_output = []
    eval_output.append(ACC);eval_output.append(SN)  #eval_output.append(AUC)
    eval_output.append(SP);eval_output.append(MCC)
    eval_output.append(score_)
    eval_output = np.array(eval_output,dtype=float)
    np.savetxt("proba.data",proba,fmt="%f",delimiter="\t")
    np.savetxt("test_y.data",test_y,fmt="%f",delimiter="\t")
    np.savetxt("predict.data",predict_,fmt="%f",delimiter="\t")
    np.savetxt("eval_output.data",eval_output,fmt="%f",delimiter="\t")
    print("Wrote results to output.data...EOF...")
    return ACC,SN,SP

def mean_fun(onelist):
    count = 0
    for i in onelist:
        count += i
    return float(count/len(onelist))

def crossValidation(clf, clfname, curdir,train_all, test_all):
    os.chdir(curdir)
    #构造出纯数据型样本集
    cur_path = curdir
    ACCs = [];SNs = []; SPs =[]
    for i in range(len(train_all)):
        os.chdir(cur_path)
        train_data = train_all[i];train_X = [];train_y = []
        test_data = test_all[i];test_X = [];test_y = []
        for eachline_train in train_data:
            one_train = eachline_train.split('\t')
            one_train_format = []
            for index in range(3,len(one_train)-1):
                one_train_format.append(float(one_train[index]))
            train_X.append(one_train_format)
            train_y.append(int(one_train[-1].strip()))
        for eachline_test in test_data:
            one_test = eachline_test.split('\t')
            one_test_format = []
            for index in range(3,len(one_test)-1):
                one_test_format.append(float(one_test[index]))
            test_X.append(one_test_format)
            test_y.append(int(one_test[-1].strip()))
        #======================================================================
        #classifier start here
        if not os.path.exists(clfname):#使用的分类器
            os.mkdir(clfname)
        out_path = clfname + "/" + clfname + "_00" + str(i)#计算结果文件夹
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        os.chdir(out_path)
        ACC, SN, SP = classifier(clf, train_X, train_y, test_X, test_y)
        ACCs.append(ACC);SNs.append(SN);SPs.append(SP)
        #======================================================================
    ACC_mean = mean_fun(ACCs)
    SN_mean = mean_fun(SNs)
    SP_mean = mean_fun(SPs)
    #==========================================================================
    #output experiment result
    os.chdir("../")
    os.system("echo `date` '" + str(clf) + "' >> log.out")
    os.system("echo ACC_mean=" + str(ACC_mean) + " >> log.out")
    os.system("echo SN_mean=" + str(SN_mean) + " >> log.out")
    os.system("echo SP_mean=" + str(SP_mean) + " >> log.out")
    return ACC_mean, SN_mean, SP_mean

cv = cross_validation.KFold(len(labels), n_folds=10)

ACCs = [];SNs = []; SPs =[]

for train_index, test_index in cv:
    print('TRAIN: ', train_index, "TEST: ", test_index)
    x_train, x_test = group[train_index], group[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    print(x_train, x_test, y_train, y_test)

    clf = RandomForestClassifier()
    ACC, SN, SP = classifier(clf, x_train, y_train, x_test, y_test)
    ACCs.append(ACC);SNs.append(SN);SPs.append(SP)

