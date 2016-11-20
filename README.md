# K近邻分类器交叉验证 K-NearestNeighbor Classifier & Cross validation
- Scarb

[TOC]

## 1. 要求描述
1. 实现K近邻分类器
2. 应用于数据集，计算10倍交叉验证`10-fold cross-validation`的分类精度
3. 计算计算留一法交叉验证`Leave-One-Out cross-validation`的分类精度

## 2. 运行环境
- python 3.5+

所需python库：
- numpy
- scipy
- xlrd
- sklearn

## 3. 解决思路
将K近邻算法用做分类器，然后在交叉验证中使用K近邻分类器进行验证。

十折交叉验证：

1. 每次取1/10的数据作为测试，其他用做训练。
2. 用KNN算法分别预测这些数据的类型
3. 将预测的类型与实际真实类型作对比，并算出正确率
4. 取另外1/10的数据，其他用做训练，循环2-4步骤
5. 将之前计算的正确率取平均值

## 4. K近邻分类器 KNN
### 4.1 KNN原理
![KNN_Origin][img1]
根据上图所示，有两类不同的样本数据，分别用蓝色的小正方形和红色的小三角形表示，而图正中间的那个绿色的圆所标示的数据则是待分类的数据。也就是说，现在， 我们不知道中间那个绿色的数据是从属于哪一类（蓝色小正方形or红色小三角形），下面，我们就要解决这个问题：给这个绿色的圆分类。

　　我们常说，物以类聚，人以群分，判别一个人是一个什么样品质特征的人，常常可以从他or她身边的朋友入手，所谓观其友，而识其人。我们不是要判别上图中那个绿色的圆是属于哪一类数据么，好说，从它的邻居下手。但一次性看多少个邻居呢？从上图中，你还能看到：

- 如果K=3，绿色圆点的最近的3个邻居是2个红色小三角形和1个蓝色小正方形，少数从属于多数，基于统计的方法，判定绿色的这个待分类点属于红色的三角形一类。
- 如果K=5，绿色圆点的最近的5个邻居是2个红色三角形和3个蓝色的正方形，还是少数从属于多数，基于统计的方法，判定绿色的这个待分类点属于蓝色的正方形一类。

于此我们看到，当无法判定当前待分类点是从属于已知分类中的哪一类时，我们可以依据统计学的理论看它所处的位置特征，衡量它周围邻居的权重，而把它归为(或分配)到权重更大的那一类。这就是K近邻算法的核心思想。

KNN算法中，所选择的邻居都是已经正确分类的对象。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。

### 4.2 算法描述
算法伪码
![KNN_Alg][img2]

### 4.3 KNN实现
~~~ python
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
~~~
实现了KNN分类器，用predict函数可以根据一组testX使用KNN算法返回一组预测type值
用accuracy_score算法可以根据预测的type值与真实type值

## 5. 交叉验证 Cross validation
### 5.1 算法描述
>10折交叉验证(10-fold cross validation)，将数据集分成十份，轮流将其中9份做训练1份做验证，10次的结果的均值作为对算法精度的估计，一般还需要进行多次10折交叉验证求均值，例如：10次10折交叉验证，以求更精确一点。
交叉验证有时也称为交叉比对，如：10折交叉比对


>K折交叉验证：初始采样分割成K个子样本，一个单独的子样本被保留作为验证模型的数据，其他K-1个样本用来训练。交叉验证重复K次，每个子样本验证一次，平均K次的结果或者使用其它结合方式，最终得到一个单一估测。这个方法的优势在于，同时重复运用随机产生的子样本进行训练和验证，每次的结果验证一次，10折交叉验证是最常用的。

>留一验证：正如名称所建议， 留一验证（LOOCV）意指只使用原本样本中的一项来当做验证资料， 而剩余的则留下来当做训练资料。 这个步骤一直持续到每个样本都被当做一次验证资料。 事实上，这等同于 K-fold 交叉验证是一样的，其中K为原本样本个数。 在某些情况下是存在有效率的演算法，如使用kernel regression 和Tikhonov regularization。

### 5.2 算法伪码
~~~ java
Step1: 	将学习样本空间 C 分为大小相等的 K 份  
Step2: 	for i = 1 to K ：
			取第i份作为测试集
			for j = 1 to K:
				if i != j:
					将第j份加到训练集中，作为训练集的一部分
				end if
			end for
		end for
Step3: 	for i in (K-1训练集)：
			训练第i个训练集，得到一个分类模型
			使用该模型在第N个数据集上测试，计算并保存模型评估指标
		end for
Step4: 	计算模型的平均性能
Step5: 	用这K个模型在最终验证集的分类准确率平均值作为此K-CV下分类器的性能指标.
~~~

### 5.3 算法实现
利用sklearn库中带有的迭代器，将测试数据按照验证方式分好，使用KNN分类器进行精确度验证。

~~~ python
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
~~~

## 6. 运行结果
运行结果图片：
![result][img3]

10-fold cross validation accuracy:  0.45666666666666667

Leave-One-Out validation accuracy:  0.5336477987421383


## Reference:
【1】[机器学习算法-K最近邻从原理到实现](http://www.csuldw.com/2015/05/21/2015-05-21-KNN/)

【2】[机器学习-Cross Validation交叉验证Python实现](http://www.csuldw.com/2015/07/28/2015-07-28%20crossvalidation/)

[img1]: http://115.28.48.229/wordpress/wp-content/uploads/2016/11/KNN_Origin.png
[img2]: http://115.28.48.229/wordpress/wp-content/uploads/2016/11/KNN_Alg.png
[img3]: http://115.28.48.229/wordpress/wp-content/uploads/2016/11/KNN_Result.png