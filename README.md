# K-NearestNeighbor Classifier & Cross validation

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

## 4. K近邻分类器
### 4.1 KNN原理
![KNN_Origin][img1]
根据上图所示，有两类不同的样本数据，分别用蓝色的小正方形和红色的小三角形表示，而图正中间的那个绿色的圆所标示的数据则是待分类的数据。也就是说，现在， 我们不知道中间那个绿色的数据是从属于哪一类（蓝色小正方形or红色小三角形），下面，我们就要解决这个问题：给这个绿色的圆分类。

　　我们常说，物以类聚，人以群分，判别一个人是一个什么样品质特征的人，常常可以从他or她身边的朋友入手，所谓观其友，而识其人。我们不是要判别上图中那个绿色的圆是属于哪一类数据么，好说，从它的邻居下手。但一次性看多少个邻居呢？从上图中，你还能看到：

- 如果K=3，绿色圆点的最近的3个邻居是2个红色小三角形和1个蓝色小正方形，少数从属于多数，基于统计的方法，判定绿色的这个待分类点属于红色的三角形一类。
- 如果K=5，绿色圆点的最近的5个邻居是2个红色三角形和3个蓝色的正方形，还是少数从属于多数，基于统计的方法，判定绿色的这个待分类点属于蓝色的正方形一类。

于此我们看到，当无法判定当前待分类点是从属于已知分类中的哪一类时，我们可以依据统计学的理论看它所处的位置特征，衡量它周围邻居的权重，而把它归为(或分配)到权重更大的那一类。这就是K近邻算法的核心思想。

KNN算法中，所选择的邻居都是已经正确分类的对象。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。

### 4.2 算法描述
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

## 运行结果
![result][img]
10-fold cross validation accuracy:  0.45666666666666667
Leave-One-Out validation accuracy:  0.5336477987421383