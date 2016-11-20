from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class cross_validation:
    def __init__(self):
        return
    def classifier(self, clf,train_X, train_y, test_X, test_y): #X:训练特征，y:训练标号
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
        #==========================================================================
        return ACC

    def mean_fun(self, onelist):
        count = 0
        for i in onelist:
            count += i
        return float(count/len(onelist))